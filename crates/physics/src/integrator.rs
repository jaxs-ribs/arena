//! # Physics Integration
//! 
//! This module handles the numerical integration of physics bodies,
//! including position updates, velocity calculations, and force application.

use crate::types::{Vec3, Sphere, BoxBody, Cylinder};

/// Integration constants
const DAMPING_FACTOR: f32 = 0.999; // Slight damping to improve stability

/// Integrate sphere positions and velocities using Verlet integration
pub fn integrate_spheres(spheres: &mut [Sphere], gravity: Vec3, dt: f32) {
    for sphere in spheres.iter_mut() {
        // Apply gravity force
        let acceleration = gravity;
        
        // Simple Euler integration
        sphere.vel += acceleration * dt;
        sphere.pos += sphere.vel * dt;
        
        // Apply damping
        sphere.vel *= DAMPING_FACTOR;
    }
}

/// Integrate box positions and velocities
pub fn integrate_boxes(boxes: &mut [BoxBody], gravity: Vec3, dt: f32) {
    for box_body in boxes.iter_mut() {
        // Apply gravity force
        let acceleration = gravity;
        
        // Simple Euler integration for now
        box_body.vel += acceleration * dt;
        box_body.pos += box_body.vel * dt;
        
        // Apply damping
        box_body.vel *= DAMPING_FACTOR;
    }
}

/// Integrate cylinder positions and velocities
pub fn integrate_cylinders(cylinders: &mut [Cylinder], gravity: Vec3, dt: f32) {
    for cylinder in cylinders.iter_mut() {
        // Apply gravity force
        let acceleration = gravity;
        
        // Simple Euler integration for now
        cylinder.vel += acceleration * dt;
        cylinder.pos += cylinder.vel * dt;
        
        // Apply damping
        cylinder.vel *= DAMPING_FACTOR;
    }
}

/// Apply external forces to spheres
pub fn apply_forces_to_spheres(spheres: &mut [Sphere], forces: &[[f32; 2]], dt: f32) {
    for (i, sphere) in spheres.iter_mut().enumerate() {
        if i < forces.len() {
            let force = Vec3::new(forces[i][0], 0.0, forces[i][1]);
            let acceleration = force / sphere.mass;
            sphere.vel += acceleration * dt;
        }
    }
}