//! # Physics Simulation Core
//!
//! This module provides the main physics simulation structure and high-level
//! execution methods. It coordinates between different subsystems like
//! integration, collision detection, and constraint solving.

use crate::types::{
    BoundingBox, BoxBody, Cylinder, Joint, JointParams, RevoluteJoint,
    PrismaticJoint, BallJoint, FixedJoint, PhysParams, Plane,
    Sphere, SpatialGrid, Vec3, Vec2, Material, PhysicsDebugInfo, SpatialGridDebugInfo,
    ForceDebugInfo, VelocityDebugInfo,
};
use crate::collision::{
    update_spatial_grid, get_potential_collision_pairs,
    detect_sphere_sphere_collision, resolve_sphere_sphere_collision,
    detect_sphere_plane_collision, resolve_sphere_plane_collision,
    detect_sphere_box_collision, resolve_sphere_box_collision,
    detect_sphere_cylinder_collision, resolve_sphere_cylinder_collision,
};
use crate::integrator::{
    integrate_spheres, integrate_boxes, integrate_cylinders,
    apply_forces_to_spheres,
};
use crate::gpu_executor::execute_gpu_step;
use compute::ComputeBackend;
use std::sync::Arc;

/// Simulation error types
#[derive(Debug)]
pub enum PhysicsError {
    /// GPU backend error
    BackendError(compute::ComputeError),
    /// No spheres in simulation
    NoSpheres,
}

impl From<compute::ComputeError> for PhysicsError {
    fn from(err: compute::ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

/// Final simulation state (for testing)
#[derive(Clone, Copy, Debug)]
pub struct SphereState {
    pub pos: Vec3,
}

/// Main physics simulation container
pub struct PhysicsSim {
    // Rigid bodies
    pub spheres: Vec<Sphere>,
    pub boxes: Vec<BoxBody>,
    pub cylinders: Vec<Cylinder>,
    pub planes: Vec<Plane>,
    
    // Simulation parameters
    pub params: PhysParams,
    
    // Constraints
    pub joints: Vec<Joint>,
    pub revolute_joints: Vec<RevoluteJoint>,
    pub prismatic_joints: Vec<PrismaticJoint>,
    pub ball_joints: Vec<BallJoint>,
    pub fixed_joints: Vec<FixedJoint>,
    pub joint_params: JointParams,
    
    // Spatial acceleration
    pub spatial_grid: SpatialGrid,
    
    // Compute backend
    pub(crate) backend: Arc<dyn ComputeBackend>,
}

impl PhysicsSim {
    /// Create a new empty simulation
    pub fn new() -> Self {
        let bounds = BoundingBox {
            min: Vec3::new(-50.0, -10.0, -50.0),
            max: Vec3::new(50.0, 90.0, 50.0),
        };
        let spatial_grid = SpatialGrid::new(4.0, bounds);
        
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

    /// Create simulation with single sphere (for testing)
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let mut sim = Self::new();
        sim.add_sphere(Vec3::new(0.0, initial_height, 0.0), Vec3::ZERO, 1.0);
        sim
    }

    /// Set external force on a body
    pub fn set_force(&mut self, body_index: usize, force: [f32; 2]) {
        if body_index < self.params.forces.len() {
            self.params.forces[body_index] = force;
        }
    }

    /// Set compute backend
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = backend;
    }

    /// Configure spatial grid parameters
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

    /// Execute one physics step on GPU
    pub fn step_gpu(&mut self) -> Result<(), PhysicsError> {
        execute_gpu_step(self)?;
        Ok(())
    }

    /// Execute one physics step on CPU
    pub fn step_cpu(&mut self) {
        let dt = self.params.dt;
        
        // 1. Apply forces and integrate positions
        self.integrate_bodies_cpu(dt);
        
        // 2. Update spatial grid
        self.update_broad_phase();
        
        // 3. Detect and resolve collisions
        self.detect_and_resolve_collisions_cpu();
        
        // 4. Solve constraints
        self.solve_constraints_cpu();
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
    /// Integrate all rigid body positions and velocities
    fn integrate_bodies_cpu(&mut self, dt: f32) {
        // Apply external forces to spheres
        apply_forces_to_spheres(&mut self.spheres, &self.params.forces, dt);
        
        // Integrate each body type
        integrate_spheres(&mut self.spheres, self.params.gravity, dt);
        integrate_boxes(&mut self.boxes, self.params.gravity, dt);
        integrate_cylinders(&mut self.cylinders, self.params.gravity, dt);
    }

    /// Update spatial partitioning for broad-phase collision detection
    fn update_broad_phase(&mut self) {
        update_spatial_grid(&mut self.spatial_grid, &self.spheres);
    }

    /// Detect and resolve all collisions
    fn detect_and_resolve_collisions_cpu(&mut self) {
        // Get potential collision pairs from spatial grid
        let potential_pairs = get_potential_collision_pairs(&self.spatial_grid);
        
        // Check sphere-sphere collisions
        for (i, j) in potential_pairs {
            let (spheres_before, spheres_after) = self.spheres.split_at_mut(j);
            if let Some(contact) = detect_sphere_sphere_collision(
                &spheres_before[i],
                &spheres_after[0],
            ) {
                resolve_sphere_sphere_collision(
                    &mut spheres_before[i],
                    &mut spheres_after[0],
                    &contact,
                );
            }
        }
        
        // Check sphere-plane collisions
        for sphere in &mut self.spheres {
            for plane in &self.planes {
                if let Some(contact) = detect_sphere_plane_collision(sphere, plane) {
                    resolve_sphere_plane_collision(sphere, plane, &contact);
                }
            }
        }
        
        // Check sphere-box collisions
        for sphere in &mut self.spheres {
            for box_body in &mut self.boxes {
                if let Some(contact) = detect_sphere_box_collision(sphere, box_body) {
                    resolve_sphere_box_collision(sphere, box_body, &contact);
                }
            }
        }
        
        // Check sphere-cylinder collisions
        for sphere in &mut self.spheres {
            for cylinder in &mut self.cylinders {
                if let Some(contact) = detect_sphere_cylinder_collision(sphere, cylinder) {
                    resolve_sphere_cylinder_collision(sphere, cylinder, &contact);
                }
            }
        }
    }

    /// Solve joint constraints
    fn solve_constraints_cpu(&mut self) {
        // Solve simple distance constraints
        for joint in &self.joints {
            let (body_a, body_b) = (joint.body_a as usize, joint.body_b as usize);
            if body_a < self.spheres.len() && body_b < self.spheres.len() {
                // Get positions first to avoid borrow issues
                let pos_b = self.spheres[body_b].pos;
                let mass_b = self.spheres[body_b].mass;
                
                if let Some(sphere_a) = self.spheres.get_mut(body_a) {
                    solve_distance_constraint_one_sided(sphere_a, pos_b, mass_b, joint.rest_length);
                }
            }
        }
        
        // TODO: Implement other joint types when ready
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
        let mass = calculate_box_mass(half_extents, 1.0); // Default density
        let box_body = BoxBody {
            pos,
            half_extents,
            vel,
            mass,
            material: Material::default(),
        };
        self.boxes.push(box_body);
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
        let mass = calculate_cylinder_mass(radius, half_height * 2.0, 1.0); // Default density
        let cylinder = Cylinder {
            pos,
            vel,
            radius,
            half_height,
            mass,
            material: Material::default(),
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
    /// Add a simple distance constraint between two spheres
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        self.joints.push(Joint {
            body_a,
            body_b,
            rest_length,
            _padding: 0,
        });
    }

    /// Add a revolute (hinge) joint between two bodies
    pub fn add_revolute_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        let joint = RevoluteJoint {
            body_a: body_a_index,
            body_b: body_b_index,
            anchor_a: anchor,
            anchor_b: anchor,
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

    /// Add a prismatic (sliding) joint between two bodies
    pub fn add_prismatic_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        let joint = PrismaticJoint {
            body_a: body_a_index,
            body_b: body_b_index,
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
        };
        self.prismatic_joints.push(joint);
        self.prismatic_joints.len() - 1
    }

    /// Add a ball joint (3DOF rotation) between two bodies
    pub fn add_ball_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
    ) -> usize {
        let joint = BallJoint {
            body_a: body_a_index,
            body_b: body_b_index,
            anchor_a: anchor,
            anchor_b: anchor,
            _pad: [0.0; 2],
        };
        self.ball_joints.push(joint);
        self.ball_joints.len() - 1
    }

    /// Add a fixed joint (no relative motion) between two bodies
    pub fn add_fixed_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        relative_position: Vec3,
        relative_orientation: [f32; 4], // Quaternion
    ) -> usize {
        let joint = FixedJoint {
            body_a: body_a_index,
            body_b: body_b_index,
            anchor_a: relative_position,
            anchor_b: Vec3::ZERO,
            relative_rotation: relative_orientation,
        };
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