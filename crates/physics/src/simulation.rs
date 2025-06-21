//! # Physics Simulation Core
//!
//! This module provides the main physics simulation structure and high-level
//! execution methods. It coordinates between different subsystems like
//! integration, collision detection, and constraint solving.

use crate::types::{
    BoundingBox, BoxBody, Cylinder, Joint, JointParams, RevoluteJoint,
    PrismaticJoint, BallJoint, FixedJoint, PhysParams, Plane,
    Sphere, SpatialGrid, Vec3, Vec2, Material, PhysicsDebugInfo, SpatialGridDebugInfo,
    ForceDebugInfo, VelocityDebugInfo, BodyType,
};
use crate::collision::{
    detect_sphere_sphere_collision, resolve_sphere_sphere_collision,
    detect_sphere_plane_collision, resolve_sphere_plane_collision,
    detect_sphere_box_collision, resolve_sphere_box_collision,
    detect_sphere_cylinder_collision, resolve_sphere_cylinder_collision,
    detect_box_plane_collision, resolve_box_plane_collision,
    detect_cylinder_plane_collision, resolve_cylinder_plane_collision,
};
use crate::integrator::{
    integrate_spheres, integrate_boxes, integrate_cylinders,
    apply_forces_to_spheres, apply_forces_to_boxes,
};
use crate::gpu_executor::execute_gpu_step;
use compute::ComputeBackend;
use glam::Quat;
use std::sync::Arc;

/// Physics simulation error types.
#[derive(Debug)]
pub enum PhysicsError {
    /// GPU computation backend failed
    BackendError(compute::ComputeError),
    /// Attempted to run simulation without any spheres
    NoSpheres,
}

impl From<compute::ComputeError> for PhysicsError {
    fn from(err: compute::ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

/// Snapshot of sphere position after simulation run.
#[derive(Clone, Copy, Debug)]
pub struct SphereState {
    pub pos: Vec3,
}

/// Physics simulation orchestrator managing bodies, constraints, and spatial data.
pub struct PhysicsSim {
    // Dynamic rigid bodies
    pub spheres: Vec<Sphere>,
    pub boxes: Vec<BoxBody>,
    pub cylinders: Vec<Cylinder>,
    
    // Static collision geometry
    pub planes: Vec<Plane>,
    
    // Simulation configuration
    pub params: PhysParams,
    
    // Physical constraints
    pub joints: Vec<Joint>,
    pub revolute_joints: Vec<RevoluteJoint>,
    pub prismatic_joints: Vec<PrismaticJoint>,
    pub ball_joints: Vec<BallJoint>,
    pub fixed_joints: Vec<FixedJoint>,
    pub joint_params: JointParams,
    
    // Spatial acceleration structure
    pub spatial_grid: SpatialGrid,
    
    // GPU computation backend
    pub(crate) backend: Arc<dyn ComputeBackend>,
}

impl PhysicsSim {
    /// Create empty physics simulation with default parameters.
    pub fn new() -> Self {
        let simulation_bounds = create_default_simulation_bounds();
        let spatial_grid = create_spatial_grid_with_bounds(simulation_bounds);
        
        Self {
            spheres: Vec::new(),
            boxes: Vec::new(),
            cylinders: Vec::new(),
            planes: Vec::new(),
            params: PhysParams {
                gravity: Vec3::new(0.0, -9.81, 0.0),
                dt: 0.01,
                forces: Vec::new(),
            },
            joints: Vec::new(),
            revolute_joints: Vec::new(),
            prismatic_joints: Vec::new(),
            ball_joints: Vec::new(),
            fixed_joints: Vec::new(),
            joint_params: JointParams {
                compliance: 0.0,
                _pad: [0.0; 3],
            },
            spatial_grid,
            backend: compute::default_backend(),
        }
    }

    /// Create test simulation with single falling sphere.
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let mut simulation = Self::new();
        let position = Vec3::new(0.0, initial_height, 0.0);
        let velocity = Vec3::ZERO;
        let radius = 1.0;
        simulation.add_sphere(position, velocity, radius);
        simulation
    }

    /// Apply external force to specific body.
    pub fn set_force(&mut self, body_index: usize, force: [f32; 2]) {
        if self.is_valid_body_index(body_index) {
            self.params.forces[body_index] = force;
        }
    }
    
    fn is_valid_body_index(&self, index: usize) -> bool {
        index < self.params.forces.len()
    }

    /// Set compute backend
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = backend;
    }

    /// Configure spatial acceleration structure.
    pub fn configure_spatial_grid(&mut self, cell_size: f32, bounds: BoundingBox) {
        self.spatial_grid = SpatialGrid::new(cell_size, bounds);
    }

    /// Get spatial grid statistics
    pub fn spatial_grid_stats(&self) -> (usize, usize, f32) {
        let stats = self.spatial_grid.get_stats();
        (stats.occupied_cells, stats.total_entries, stats.average_entries_per_cell)
    }

    /// Get debug information
    pub fn get_debug_info(&self) -> PhysicsDebugInfo {
        let grid_stats = self.spatial_grid.get_stats();
        
        PhysicsDebugInfo {
            num_spheres: self.spheres.len(),
            num_boxes: self.boxes.len(),
            num_cylinders: self.cylinders.len(),
            num_planes: self.planes.len(),
            num_joints: self.joints.len(),
            gravity: self.params.gravity,
            dt: self.params.dt,
            spatial_grid: SpatialGridDebugInfo {
                cell_size: self.spatial_grid.cell_size,
                bounds: self.spatial_grid.bounds,
                occupied_cells: grid_stats.occupied_cells,
                total_entries: grid_stats.total_entries,
                average_entries_per_cell: grid_stats.average_entries_per_cell,
            },
            forces: self.spheres.iter().enumerate().map(|(i, sphere)| {
                ForceDebugInfo {
                    body_index: i,
                    applied_force: if i < self.params.forces.len() {
                        Vec3::new(self.params.forces[i][0], 0.0, self.params.forces[i][1])
                    } else {
                        Vec3::ZERO
                    },
                    gravity_force: self.params.gravity * sphere.mass,
                }
            }).collect(),
            velocities: self.spheres.iter().enumerate().map(|(i, sphere)| {
                VelocityDebugInfo {
                    body_index: i,
                    linear_velocity: sphere.vel,
                    speed: sphere.vel.length(),
                }
            }).collect(),
        }
    }

    /// Execute single physics timestep on GPU.
    pub fn step_gpu(&mut self) -> Result<(), PhysicsError> {
        execute_gpu_step(self)?;
        Ok(())
    }

    /// Execute single physics timestep on CPU.
    pub fn step_cpu(&mut self) {
        let timestep = self.params.dt;
        
        // Apply forces and integrate velocities/positions
        self.apply_forces_and_integrate(timestep);
        
        // CRITICAL: Enforce constraints AFTER integration to fix any drift
        self.solve_physical_constraints();
        
        self.update_spatial_acceleration_structure();
        self.detect_and_resolve_all_collisions();
        self.apply_2d_constraints();
    }

    /// Run simulation for multiple steps (GPU)
    pub fn run(&mut self, dt: f32, steps: usize) -> Result<SphereState, PhysicsError> {
        if self.spheres.is_empty() {
            return Err(PhysicsError::NoSpheres);
        }
        
        self.params.dt = dt;
        for _ in 0..steps {
            self.step_gpu()?;
        }
        
        Ok(SphereState {
            pos: self.spheres[0].pos,
        })
    }

    /// Run simulation for multiple steps (CPU)
    pub fn run_cpu(&mut self, dt: f32, steps: usize) {
        self.params.dt = dt;
        for _ in 0..steps {
            self.step_cpu();
        }
    }
}

// CPU simulation implementation
impl PhysicsSim {
    fn apply_forces_and_integrate(&mut self, timestep: f32) {
        apply_forces_to_spheres(&mut self.spheres, &self.params.forces, timestep);
        apply_forces_to_boxes(&mut self.boxes, &self.params.forces, timestep);
        
        integrate_spheres(&mut self.spheres, self.params.gravity, timestep);
        integrate_boxes(&mut self.boxes, self.params.gravity, timestep);
        integrate_cylinders(&mut self.cylinders, self.params.gravity, timestep);
    }

    fn update_spatial_acceleration_structure(&mut self) {
        self.spatial_grid.update(&self.spheres);
    }

    fn detect_and_resolve_all_collisions(&mut self) {
        self.resolve_sphere_sphere_collisions();
        self.resolve_sphere_static_collisions();
        self.resolve_sphere_dynamic_collisions();
    }
    
    fn resolve_sphere_sphere_collisions(&mut self) {
        let sphere_count = self.spheres.len();
        
        for i in 0..sphere_count {
            for j in (i + 1)..sphere_count {
                self.check_and_resolve_sphere_pair(i, j);
            }
        }
    }
    
    fn check_and_resolve_sphere_pair(&mut self, index_a: usize, index_b: usize) {
        let (first_part, second_part) = self.spheres.split_at_mut(index_b);
        let sphere_a = &mut first_part[index_a];
        let sphere_b = &mut second_part[0];
        
        if let Some(contact) = detect_sphere_sphere_collision(sphere_a, sphere_b) {
            resolve_sphere_sphere_collision(sphere_a, sphere_b, &contact);
        }
    }
    
    fn resolve_sphere_static_collisions(&mut self) {
        // Sphere-plane collisions
        for sphere in &mut self.spheres {
            for plane in &self.planes {
                if let Some(contact) = detect_sphere_plane_collision(sphere, plane) {
                    resolve_sphere_plane_collision(sphere, plane, &contact);
                }
            }
        }
        
        // Box-plane collisions
        for box_body in &mut self.boxes {
            for plane in &self.planes {
                if let Some(contact) = detect_box_plane_collision(box_body, plane) {
                    resolve_box_plane_collision(box_body, plane, &contact, self.params.dt);
                }
            }
        }
        
        // Cylinder-plane collisions
        for cylinder in &mut self.cylinders {
            for plane in &self.planes {
                if let Some(contact) = detect_cylinder_plane_collision(cylinder, plane) {
                    resolve_cylinder_plane_collision(cylinder, plane, &contact, self.params.dt);
                }
            }
        }
    }
    
    fn resolve_sphere_dynamic_collisions(&mut self) {
        for sphere in &mut self.spheres {
            for box_body in &mut self.boxes {
                if let Some(contact) = detect_sphere_box_collision(sphere, box_body) {
                    resolve_sphere_box_collision(sphere, box_body, &contact);
                }
            }
            
            for cylinder in &mut self.cylinders {
                if let Some(contact) = detect_sphere_cylinder_collision(sphere, cylinder) {
                    resolve_sphere_cylinder_collision(sphere, cylinder, &contact);
                }
            }
        }
    }

    fn solve_physical_constraints(&mut self) {
        self.solve_distance_joint_constraints();
        self.solve_revolute_joint_constraints();
        // Future: Add prismatic and other constraint types
    }
    
    fn apply_2d_constraints(&mut self) {
        // Constrain boxes that are part of revolute joints to X-Y plane
        for joint in &self.revolute_joints {
            let box_idx = joint.body_a as usize;
            let cylinder_idx = joint.body_b as usize;
            
            // Constrain box to 2D plane if it's part of a revolute joint
            if box_idx < self.boxes.len() {
                self.boxes[box_idx].pos.z = 0.0;
                self.boxes[box_idx].vel.z = 0.0;
                // Constrain rotation to Z-axis only
                self.boxes[box_idx].angular_vel.x = 0.0;
                self.boxes[box_idx].angular_vel.y = 0.0;
            }
            
            // Constrain cylinder to 2D plane
            if cylinder_idx < self.cylinders.len() {
                self.cylinders[cylinder_idx].pos.z = 0.0;
                self.cylinders[cylinder_idx].vel.z = 0.0;
                // Keep only Z-axis rotation
                self.cylinders[cylinder_idx].angular_vel.x = 0.0;
                self.cylinders[cylinder_idx].angular_vel.y = 0.0;
            }
        }
    }
    
    fn solve_distance_joint_constraints(&mut self) {
        let joints = self.joints.clone();
        for joint in &joints {
            let body_a_index = joint.body_a as usize;
            let body_b_index = joint.body_b as usize;
            
            if self.are_valid_sphere_indices(body_a_index, body_b_index) {
                self.apply_distance_constraint(body_a_index, body_b_index, joint.rest_length);
            }
        }
    }
    
    fn are_valid_sphere_indices(&self, index_a: usize, index_b: usize) -> bool {
        index_a < self.spheres.len() && index_b < self.spheres.len()
    }
    
    fn apply_distance_constraint(&mut self, body_a_index: usize, body_b_index: usize, rest_length: f32) {
        let sphere_b_position = self.spheres[body_b_index].pos;
        let sphere_b_mass = self.spheres[body_b_index].mass;
        
        if let Some(sphere_a) = self.spheres.get_mut(body_a_index) {
            solve_distance_constraint_one_sided(sphere_a, sphere_b_position, sphere_b_mass, rest_length);
        }
    }
    
    fn solve_revolute_joint_constraints(&mut self) {
        // For now, implement a simple version that maintains the anchor points together
        // Full angular constraints will be added next
        let joints = self.revolute_joints.clone();
        
        for joint in &joints {
            let body_a_idx = joint.body_a as usize;
            let body_b_idx = joint.body_b as usize;
            
            // For now, only support box-cylinder connections
            if body_a_idx < self.boxes.len() && body_b_idx < self.cylinders.len() {
                self.solve_revolute_constraint_box_cylinder(body_a_idx, body_b_idx, joint);
            }
        }
    }
    
    fn solve_revolute_constraint_box_cylinder(
        &mut self, 
        box_idx: usize, 
        cylinder_idx: usize, 
        joint: &RevoluteJoint
    ) {
        // Get cart state
        let cart_pos = self.boxes[box_idx].pos;
        let cart_vel = self.boxes[box_idx].vel;
        
        // Get the joint position in world space (top of cart)
        let joint_world_pos = cart_pos + joint.anchor_a;
        
        // Physics parameters
        let gravity_magnitude = self.params.gravity.length();
        let mass = self.cylinders[cylinder_idx].mass;
        let pole_length = self.cylinders[cylinder_idx].half_height * 2.0;
        let lever_arm = pole_length / 2.0; // Distance from joint to center of mass
        
        // Get current pole position and calculate angle from it
        let pole_pos = self.cylinders[cylinder_idx].pos;
        let pole_offset = pole_pos - joint_world_pos;
        
        // Calculate current angle from actual pole position
        let current_angle = pole_offset.x.atan2(pole_offset.y);
        
        // 1. Gravity torque for INVERTED pendulum (pole standing UP)
        let gravity_torque = mass * gravity_magnitude * lever_arm * current_angle.sin();
        
        // 2. Cart acceleration effect
        let cart_acceleration = if self.boxes[box_idx].body_type == BodyType::Kinematic {
            cart_vel.x * 5.0
        } else {
            0.0
        };
        let acceleration_torque = -mass * cart_acceleration * lever_arm * current_angle.cos();
        
        // 3. Total torque
        let total_torque = gravity_torque + acceleration_torque;
        
        // 4. Moment of inertia for rod about end
        let moment_of_inertia = (1.0 / 3.0) * mass * pole_length * pole_length;
        
        // 5. Angular acceleration
        let angular_acceleration = total_torque / moment_of_inertia;
        
        // 6. Update angular velocity
        self.cylinders[cylinder_idx].angular_vel.z += angular_acceleration * self.params.dt;
        
        // 7. Apply damping
        self.cylinders[cylinder_idx].angular_vel.z *= 0.998;
        
        // 8. Update angle
        let new_angle = current_angle + self.cylinders[cylinder_idx].angular_vel.z * self.params.dt;
        
        // 9. Update orientation quaternion for rendering
        let half_angle = new_angle * 0.5;
        self.cylinders[cylinder_idx].orientation = [
            0.0,
            0.0,
            half_angle.sin(),
            half_angle.cos(),
        ];
        
        // 10. ENFORCE POSITION CONSTRAINT: pole must be attached at joint
        // Calculate where pole center MUST be based on joint position and angle
        let direction = Vec3::new(new_angle.sin(), new_angle.cos(), 0.0);
        let constrained_pole_center = joint_world_pos + direction * lever_arm;
        
        // 11. Hard constraint: directly set pole position
        self.cylinders[cylinder_idx].pos = constrained_pole_center;
        
        // 12. Calculate pole velocity consistently with constraint
        // Velocity at pole center = cart velocity + rotational component
        let angular_vel_z = self.cylinders[cylinder_idx].angular_vel.z;
        let tangent = Vec3::new(new_angle.cos(), -new_angle.sin(), 0.0);
        let rotation_velocity = tangent * (lever_arm * angular_vel_z);
        self.cylinders[cylinder_idx].vel = cart_vel + rotation_velocity;
        
        // 13. Constrain to 2D plane
        self.cylinders[cylinder_idx].pos.z = 0.0;
        self.cylinders[cylinder_idx].vel.z = 0.0;
        self.cylinders[cylinder_idx].angular_vel.x = 0.0;
        self.cylinders[cylinder_idx].angular_vel.y = 0.0;
    }
}

/// Solve a distance constraint by moving only sphere_a
fn solve_distance_constraint_one_sided(
    sphere_a: &mut Sphere, 
    pos_b: Vec3, 
    mass_b: f32, 
    rest_length: f32
) {
    let delta = pos_b - sphere_a.pos;
    let current_length = delta.length();
    
    if current_length > 0.0001 {
        let correction = delta * ((rest_length - current_length) / current_length);
        let mass_ratio = sphere_a.mass / (sphere_a.mass + mass_b);
        
        // Only move sphere_a (simplified for now)
        sphere_a.pos -= correction * (1.0 - mass_ratio);
    }
}

impl Default for PhysicsSim {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Builder Methods ====================
// Methods for adding rigid bodies and constraints to the simulation

impl PhysicsSim {
    /// Add a sphere with default material properties
    pub fn add_sphere(&mut self, pos: Vec3, vel: Vec3, radius: f32) -> usize {
        self.add_sphere_with_material(pos, vel, radius, Material::default())
    }

    /// Add a sphere with custom material properties
    pub fn add_sphere_with_material(
        &mut self, 
        pos: Vec3, 
        vel: Vec3, 
        radius: f32, 
        material: Material
    ) -> usize {
        let mass = calculate_sphere_mass(radius, material.density);
        self.add_sphere_with_mass_and_material(pos, vel, radius, mass, material)
    }

    /// Add a sphere with explicit mass and material
    pub fn add_sphere_with_mass_and_material(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        radius: f32,
        mass: f32,
        material: Material,
    ) -> usize {
        let sphere = Sphere::with_mass_and_material(pos, vel, radius, mass, material);
        self.spheres.push(sphere);
        self.params.forces.push([0.0, 0.0]);
        self.spheres.len() - 1
    }

    /// Add a box-shaped rigid body
    pub fn add_box(&mut self, pos: Vec3, half_extents: Vec3, vel: Vec3) -> usize {
        self.add_box_with_type(pos, half_extents, vel, BodyType::Dynamic)
    }
    
    /// Add a box-shaped rigid body with specific body type
    pub fn add_box_with_type(&mut self, pos: Vec3, half_extents: Vec3, vel: Vec3, body_type: BodyType) -> usize {
        let mass = calculate_box_mass(half_extents, 1.0); // Default density
        let box_body = BoxBody {
            pos,
            half_extents,
            vel,
            mass,
            orientation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            angular_vel: Vec3::ZERO,
            material: Material::default(),
            body_type,
        };
        self.boxes.push(box_body);
        self.params.forces.push([0.0, 0.0]);
        self.boxes.len() - 1
    }

    /// Add a cylindrical rigid body
    pub fn add_cylinder(
        &mut self,
        pos: Vec3,
        radius: f32,
        half_height: f32,
        vel: Vec3,
    ) -> usize {
        self.add_cylinder_with_type(pos, radius, half_height, vel, BodyType::Dynamic)
    }
    
    /// Add a cylindrical rigid body with specific body type
    pub fn add_cylinder_with_type(
        &mut self,
        pos: Vec3,
        radius: f32,
        half_height: f32,
        vel: Vec3,
        body_type: BodyType,
    ) -> usize {
        let mass = calculate_cylinder_mass(radius, half_height * 2.0, 1.0); // Default density
        let cylinder = Cylinder {
            pos,
            vel,
            radius,
            half_height,
            mass,
            orientation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            angular_vel: Vec3::ZERO,
            material: Material::default(),
            body_type,
            shape_offset: Vec3::ZERO, // Default: shape at center of mass
            mesh_offset: Vec3::ZERO,  // Default: mesh origin at center of mass
        };
        self.cylinders.push(cylinder);
        self.cylinders.len() - 1
    }
    
    /// Add a cylinder with custom shape and mesh offsets
    pub fn add_cylinder_with_offsets(
        &mut self,
        pos: Vec3,
        radius: f32,
        half_height: f32,
        vel: Vec3,
        body_type: BodyType,
        shape_offset: Vec3,
        mesh_offset: Vec3,
    ) -> usize {
        let mass = calculate_cylinder_mass(radius, half_height * 2.0, 1.0); // Default density
        let cylinder = Cylinder {
            pos,
            vel,
            radius,
            half_height,
            mass,
            orientation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            angular_vel: Vec3::ZERO,
            material: Material::default(),
            body_type,
            shape_offset,
            mesh_offset,
        };
        self.cylinders.push(cylinder);
        self.cylinders.len() - 1
    }

    /// Add a static plane for collision
    pub fn add_plane(&mut self, normal: Vec3, d: f32, extents: Vec2) -> usize {
        let plane = Plane {
            normal,
            d,
            extents,
            material: Material::default(),
        };
        self.planes.push(plane);
        self.planes.len() - 1
    }
}

// ==================== Joint Builder Methods ====================

impl PhysicsSim {
    /// Add distance constraint between two spheres.
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        let joint = create_distance_joint(body_a, body_b, rest_length);
        self.joints.push(joint);
    }

    /// Add revolute (hinge) joint between two bodies.
    pub fn add_revolute_joint(
        &mut self,
        _body_a_type: u32,
        body_a_index: u32,
        _body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        // Calculate anchor points in local coordinates for each body
        let body_a_idx = body_a_index as usize;
        let body_b_idx = body_b_index as usize;
        
        // Calculate proper anchor points for box-cylinder connection (CartPole)
        let anchor_a = if body_a_idx < self.boxes.len() {
            // For the cart (box): anchor at TOP of cart (joint connection point)
            // Cart half_extents.y is half-height, so top is at +half_extents.y
            Vec3::new(0.0, self.boxes[body_a_idx].half_extents.y, 0.0)
        } else {
            Vec3::ZERO
        };
        
        let anchor_b = if body_b_idx < self.cylinders.len() {
            // For the pole (cylinder): anchor at the BOTTOM of the pole
            // Cylinder position is at center, so bottom is at -half_height in Y
            let pole_bottom_offset = Vec3::new(0.0, -self.cylinders[body_b_idx].half_height, 0.0);
            pole_bottom_offset
        } else {
            Vec3::ZERO
        };
        
        let joint = RevoluteJoint {
            body_a: body_a_index,
            body_b: body_b_index,
            anchor_a,
            anchor_b,
            axis,
            lower_limit: -std::f32::consts::PI,
            upper_limit: std::f32::consts::PI,
            motor_speed: 0.0,
            motor_max_force: 0.0,
            enable_motor: 0,
            enable_limit: 0,
            _pad: 0.0,
        };
        self.revolute_joints.push(joint);
        self.revolute_joints.len() - 1
    }

    /// Add prismatic (sliding) joint between two bodies.
    pub fn add_prismatic_joint(
        &mut self,
        _body_a_type: u32,
        body_a_index: u32,
        _body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        let joint = create_prismatic_joint(body_a_index, body_b_index, anchor, axis);
        self.prismatic_joints.push(joint);
        self.prismatic_joints.len() - 1
    }

    /// Add ball joint (3DOF rotation) between two bodies.
    pub fn add_ball_joint(
        &mut self,
        _body_a_type: u32,
        body_a_index: u32,
        _body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
    ) -> usize {
        let joint = create_ball_joint(body_a_index, body_b_index, anchor);
        self.ball_joints.push(joint);
        self.ball_joints.len() - 1
    }

    /// Add fixed joint (no relative motion) between two bodies.
    pub fn add_fixed_joint(
        &mut self,
        _body_a_type: u32,
        body_a_index: u32,
        _body_b_type: u32,
        body_b_index: u32,
        relative_position: Vec3,
        relative_orientation: [f32; 4],
    ) -> usize {
        let joint = create_fixed_joint(
            body_a_index,
            body_b_index,
            relative_position,
            relative_orientation
        );
        self.fixed_joints.push(joint);
        self.fixed_joints.len() - 1
    }
}

// Helper functions for mass calculations
fn calculate_sphere_mass(radius: f32, density: f32) -> f32 {
    let volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    volume * density
}

fn calculate_box_mass(half_extents: Vec3, density: f32) -> f32 {
    let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
    volume * density
}

fn calculate_cylinder_mass(radius: f32, height: f32, density: f32) -> f32 {
    let volume = std::f32::consts::PI * radius.powi(2) * height;
    volume * density
}

// Simulation configuration helpers
fn create_default_simulation_bounds() -> BoundingBox {
    BoundingBox {
        min: Vec3::new(-50.0, -10.0, -50.0),
        max: Vec3::new(50.0, 90.0, 50.0),
    }
}

fn create_spatial_grid_with_bounds(bounds: BoundingBox) -> SpatialGrid {
    const DEFAULT_CELL_SIZE: f32 = 4.0;
    SpatialGrid::new(DEFAULT_CELL_SIZE, bounds)
}

fn create_default_physics_parameters() -> PhysParams {
    PhysParams {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 0.01,
        forces: Vec::new(),
    }
}

// Joint creation helpers
fn create_distance_joint(body_a: u32, body_b: u32, rest_length: f32) -> Joint {
    Joint {
        body_a,
        body_b,
        rest_length,
        _padding: 0,
    }
}


fn create_prismatic_joint(
    body_a: u32,
    body_b: u32,
    anchor: Vec3,
    axis: Vec3,
) -> PrismaticJoint {
    PrismaticJoint {
        body_a,
        body_b,
        anchor_a: anchor,
        anchor_b: anchor,
        axis,
        lower_limit: -1.0,
        upper_limit: 1.0,
        motor_speed: 0.0,
        motor_max_force: 0.0,
        enable_motor: 0,
        enable_limit: 0,
        _pad: 0.0,
    }
}

fn create_ball_joint(body_a: u32, body_b: u32, anchor: Vec3) -> BallJoint {
    BallJoint {
        body_a,
        body_b,
        anchor_a: anchor,
        anchor_b: anchor,
        _pad: [0.0; 2],
    }
}

fn create_fixed_joint(
    body_a: u32,
    body_b: u32,
    relative_position: Vec3,
    relative_orientation: [f32; 4],
) -> FixedJoint {
    FixedJoint {
        body_a,
        body_b,
        anchor_a: relative_position,
        anchor_b: Vec3::ZERO,
        relative_rotation: relative_orientation,
    }
}