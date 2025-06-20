//! CartPole entity for reinforcement learning environments
//!
//! This module provides a high-level CartPole entity that wraps
//! the physics simulation components for creating cartpole systems.

use crate::types::{Vec3, Vec2};
use crate::PhysicsSim;

/// Configuration for a CartPole entity
#[derive(Clone, Debug)]
pub struct CartPoleConfig {
    /// Cart half-extents (width, height, depth)
    pub cart_size: Vec3,
    /// Cart mass in kg
    pub cart_mass: f32,
    /// Pole length in meters
    pub pole_length: f32,
    /// Pole radius in meters
    pub pole_radius: f32,
    /// Pole mass in kg
    pub pole_mass: f32,
    /// Initial pole angle from vertical (radians)
    pub initial_angle: f32,
    /// Force magnitude that can be applied to cart
    pub force_magnitude: f32,
    /// Angle threshold for failure detection (radians)
    pub failure_angle: f32,
    /// Position threshold for failure detection (meters)
    pub position_limit: f32,
}

impl Default for CartPoleConfig {
    fn default() -> Self {
        Self {
            cart_size: Vec3::new(0.6, 0.1, 0.3),  // Much thinner cart: wider but lower height
            cart_mass: 1.0,
            pole_length: 2.0,
            pole_radius: 0.05,
            pole_mass: 0.1,
            initial_angle: 0.05, // Small initial perturbation
            force_magnitude: 10.0,
            failure_angle: 1.4, // ~80 degrees (close to horizontal)
            position_limit: 4.0, // 4 meters from center
        }
    }
}

/// A CartPole entity in the physics simulation
pub struct CartPole {
    /// Index of the cart body in the simulation
    pub cart_idx: usize,
    /// Index of the pole body in the simulation
    pub pole_idx: usize,
    /// Index of the revolute joint
    pub joint_idx: usize,
    /// Configuration for this cartpole
    pub config: CartPoleConfig,
    /// Initial position for reset
    initial_position: Vec3,
    /// Initial Z position for 2D constraints (preserve grid layout)
    pub initial_z: f32,
    /// Whether the cartpole has failed (fallen over or out of bounds)
    pub failed: bool,
}

impl CartPole {
    /// Create a new CartPole entity in the simulation
    pub fn new(sim: &mut PhysicsSim, position: Vec3, config: CartPoleConfig) -> Self {
        // Create cart as kinematic body - preserve Z position for grid layout
        let cart_pos = Vec3::new(position.x, position.y + config.cart_size.y, position.z);
        let cart_idx = sim.add_box_with_type(cart_pos, config.cart_size, Vec3::ZERO, crate::types::BodyType::Kinematic);
        sim.boxes[cart_idx].mass = config.cart_mass;
        
        // Set cart material properties
        sim.boxes[cart_idx].material.friction = 0.8;
        sim.boxes[cart_idx].material.restitution = 0.0;
        
        // Calculate joint position (top of cart)
        let joint_anchor_on_cart = Vec3::new(0.0, config.cart_size.y, 0.0);
        let joint_world_pos = cart_pos + joint_anchor_on_cart;
        
        // Position pole with initial angle (in X-Y plane only)
        let pole_half_height = config.pole_length / 2.0;
        let pole_offset_x = config.initial_angle.sin() * pole_half_height;
        let pole_offset_y = config.initial_angle.cos() * pole_half_height;
        let pole_pos = Vec3::new(
            joint_world_pos.x + pole_offset_x,
            joint_world_pos.y + pole_offset_y,
            0.0  // Always at z=0 for 2D constraint
        );
        
        // Create pole as dynamic body (affected by gravity)
        let pole_idx = sim.add_cylinder_with_type(
            pole_pos,
            config.pole_radius,
            pole_half_height,
            Vec3::ZERO,
            crate::types::BodyType::Dynamic
        );
        sim.cylinders[pole_idx].mass = config.pole_mass;
        
        // CRITICAL FIX: Set initial orientation to match the initial angle
        let half_angle = config.initial_angle * 0.5;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        sim.cylinders[pole_idx].orientation = [
            0.0,         // x
            0.0,         // y
            sin_half,    // z (rotation around z-axis)
            cos_half,    // w
        ];
        
        // Create revolute joint
        let joint_idx = sim.add_revolute_joint(
            0, cart_idx as u32,  // Box type
            2, pole_idx as u32,  // Cylinder type
            joint_world_pos,
            Vec3::new(0.0, 0.0, 1.0) // Rotate around Z axis (perpendicular to X-Y plane)
        );
        
        Self {
            cart_idx,
            pole_idx,
            joint_idx,
            config,
            initial_position: position,
            initial_z: position.z,  // Store initial Z for 2D constraints
            failed: false,
        }
    }
    
    /// Apply force to the cart (-1.0 = left, 0.0 = none, 1.0 = right)
    pub fn apply_force(&self, sim: &mut PhysicsSim, action: f32) {
        let force = action.clamp(-1.0, 1.0) * self.config.force_magnitude;
        sim.set_force(self.cart_idx, [force, 0.0]);
    }
    
    /// Check if the cartpole has failed (fallen over or out of bounds)
    pub fn check_failure(&mut self, sim: &PhysicsSim) -> bool {
        if self.failed {
            return true;
        }
        
        // Check cart position limits
        let cart_pos = sim.boxes[self.cart_idx].pos;
        if cart_pos.x.abs() > self.config.position_limit {
            self.failed = true;
            return true;
        }
        
        // Check pole angle
        let pole_angle = self.get_pole_angle(sim);
        if pole_angle.abs() > self.config.failure_angle {
            self.failed = true;
            return true;
        }
        
        false
    }
    
    /// Get the current pole angle from vertical (radians)
    pub fn get_pole_angle(&self, sim: &PhysicsSim) -> f32 {
        let cart_pos = sim.boxes[self.cart_idx].pos;
        let pole_pos = sim.cylinders[self.pole_idx].pos;
        let joint_pos = cart_pos + Vec3::new(0.0, self.config.cart_size.y, 0.0);
        
        let pole_vector = pole_pos - joint_pos;
        pole_vector.x.atan2(pole_vector.y)
    }
    
    /// Get the current state vector [cart_x, cart_vel, pole_angle, pole_angular_vel]
    pub fn get_state(&self, sim: &PhysicsSim) -> [f32; 4] {
        let cart = &sim.boxes[self.cart_idx];
        let pole = &sim.cylinders[self.pole_idx];
        
        [
            cart.pos.x,
            cart.vel.x,
            self.get_pole_angle(sim),
            pole.angular_vel.z, // Angular velocity around Z axis
        ]
    }
    
    /// Reset the cartpole to its initial state
    pub fn reset(&mut self, sim: &mut PhysicsSim) {
        self.failed = false;
        
        // Reset cart
        let cart_pos = Vec3::new(
            self.initial_position.x,
            self.initial_position.y + self.config.cart_size.y,
            self.initial_position.z
        );
        sim.boxes[self.cart_idx].pos = cart_pos;
        sim.boxes[self.cart_idx].vel = Vec3::ZERO;
        sim.boxes[self.cart_idx].angular_vel = Vec3::ZERO;
        
        // Reset pole with initial angle
        let joint_pos = cart_pos + Vec3::new(0.0, self.config.cart_size.y, 0.0);
        let pole_half_height = self.config.pole_length / 2.0;
        let pole_offset_x = self.config.initial_angle.sin() * pole_half_height;
        let pole_offset_y = self.config.initial_angle.cos() * pole_half_height;
        let pole_pos = Vec3::new(
            joint_pos.x + pole_offset_x,
            joint_pos.y + pole_offset_y,
            joint_pos.z
        );
        
        sim.cylinders[self.pole_idx].pos = pole_pos;
        sim.cylinders[self.pole_idx].vel = Vec3::ZERO;
        sim.cylinders[self.pole_idx].angular_vel = Vec3::ZERO;
        
        // CRITICAL FIX: Reset orientation to match initial angle
        let half_angle = self.config.initial_angle * 0.5;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        sim.cylinders[self.pole_idx].orientation = [
            0.0,         // x
            0.0,         // y
            sin_half,    // z (rotation around z-axis)
            cos_half,    // w
        ];
        
        // Clear any applied forces
        sim.set_force(self.cart_idx, [0.0, 0.0]);
    }
}

/// Manages multiple CartPole entities in a grid layout
pub struct CartPoleGrid {
    /// All cartpoles in the grid
    pub cartpoles: Vec<CartPole>,
    /// Grid dimensions (rows, columns)
    pub grid_size: (usize, usize),
    /// Spacing between cartpoles
    pub spacing: f32,
}

impl CartPoleGrid {
    /// Create a grid of CartPoles
    pub fn new(
        sim: &mut PhysicsSim,
        rows: usize,
        cols: usize,
        spacing: f32,
        config: CartPoleConfig,
    ) -> Self {
        let mut cartpoles = Vec::new();
        
        // Calculate grid offsets to center it, ensuring we stay within position limits
        let grid_width = (cols as f32 - 1.0) * spacing;
        let grid_depth = (rows as f32 - 1.0) * spacing;
        
        // Ensure grid doesn't exceed position limits with safety margin
        let safety_margin = 0.5;
        let max_allowed_width = (config.position_limit - safety_margin) * 2.0;
        
        if grid_width > max_allowed_width {
            panic!("Grid too wide for position limits: width={:.2}, max_allowed={:.2}. Reduce spacing or number of columns.", 
                   grid_width, max_allowed_width);
        }
        
        let start_x = -grid_width / 2.0;
        let start_z = -grid_depth / 2.0;
        
        // Create cartpoles in a single line (Z=0) to respect 2D constraints
        // Total count = rows * cols, arrange in a line along X axis
        let total_count = rows * cols;
        let total_width = (total_count as f32 - 1.0) * spacing;
        let line_start_x = -total_width / 2.0;
        
        // Ensure line fits within position limits
        let safety_margin = 0.5;
        let max_allowed_width = (config.position_limit - safety_margin) * 2.0;
        
        if total_width > max_allowed_width {
            panic!("Line of CartPoles too wide for position limits: width={:.2}, max_allowed={:.2}. Reduce spacing or total count.", 
                   total_width, max_allowed_width);
        }
        
        for i in 0..total_count {
            let x = line_start_x + i as f32 * spacing;
            let position = Vec3::new(x, 0.0, 0.0);
            
            let cartpole = CartPole::new(sim, position, config.clone());
            cartpoles.push(cartpole);
        }
        
        Self {
            cartpoles,
            grid_size: (rows, cols),
            spacing,
        }
    }
    
    /// Apply actions to all cartpoles
    pub fn apply_actions(&self, sim: &mut PhysicsSim, actions: &[f32]) {
        for (i, cartpole) in self.cartpoles.iter().enumerate() {
            if i < actions.len() {
                cartpole.apply_force(sim, actions[i]);
            }
        }
    }
    
    /// Check all cartpoles for failure and reset failed ones
    pub fn check_and_reset_failures(&mut self, sim: &mut PhysicsSim) -> Vec<usize> {
        let mut failed_indices = Vec::new();
        
        for (i, cartpole) in self.cartpoles.iter_mut().enumerate() {
            if cartpole.check_failure(sim) {
                failed_indices.push(i);
                cartpole.reset(sim);
            }
        }
        
        failed_indices
    }
    
    /// Get states of all cartpoles
    pub fn get_all_states(&self, sim: &PhysicsSim) -> Vec<[f32; 4]> {
        self.cartpoles.iter()
            .map(|cp| cp.get_state(sim))
            .collect()
    }
    
    /// Reset all cartpoles
    pub fn reset_all(&mut self, sim: &mut PhysicsSim) {
        for cartpole in &mut self.cartpoles {
            cartpole.reset(sim);
        }
    }
}