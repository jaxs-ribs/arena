//! # Physics Simulation Core
//!
//! This module provides the main physics simulation structure and high-level
//! execution methods. It coordinates between different subsystems like
//! integration, collision detection, and constraint solving.

use crate::types::{
    BoundingBox, BoxBody, Cylinder, Joint, JointParams, RevoluteJoint,
    PrismaticJoint, BallJoint, FixedJoint, PhysParams, Plane,
    Sphere, SpatialGrid, Vec3, PhysicsDebugInfo, SpatialGridDebugInfo,
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
            if let (Some(sphere_a), Some(sphere_b)) = (
                self.spheres.get_mut(joint.body_a as usize),
                self.spheres.get(joint.body_b as usize).cloned(),
            ) {
                solve_distance_constraint(sphere_a, &sphere_b, joint.rest_length);
            }
        }
        
        // TODO: Implement other joint types when ready
    }
}

/// Solve a distance constraint between two spheres
fn solve_distance_constraint(sphere_a: &mut Sphere, sphere_b: &Sphere, rest_length: f32) {
    let delta = sphere_b.pos - sphere_a.pos;
    let current_length = delta.length();
    
    if current_length > 0.0001 {
        let correction = delta * ((rest_length - current_length) / current_length);
        let mass_ratio = sphere_a.mass / (sphere_a.mass + sphere_b.mass);
        
        // Only move sphere_a (simplified for now)
        sphere_a.pos -= correction * (1.0 - mass_ratio);
    }
}

impl Default for PhysicsSim {
    fn default() -> Self {
        Self::new()
    }
}