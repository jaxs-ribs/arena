//! # Physics Integration
//! 
//! This module handles the numerical integration of physics bodies,
//! including position updates, velocity calculations, and force application.

use crate::types::{Vec3, Sphere, BoxBody, Cylinder};

/// Integration constants
const DAMPING_FACTOR: f32 = 1.0; // No damping for now (was 0.999)

/// Integrate sphere positions and velocities using Verlet integration
pub fn integrate_spheres(spheres: &mut [Sphere], gravity: Vec3, dt: f32) {
    for sphere in spheres.iter_mut() {
        // Apply gravity force
        let acceleration = gravity;
        
        // Simple Euler integration
        sphere.vel += acceleration * dt;
        sphere.pos += sphere.vel * dt;
        
        // Apply damping (disabled for now)
        // sphere.vel *= DAMPING_FACTOR;
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
        
        // Apply damping (disabled for now)
        // box_body.vel *= DAMPING_FACTOR;
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
        
        // Apply damping (disabled for now)  
        // cylinder.vel *= DAMPING_FACTOR;
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

/// Apply external forces to boxes
pub fn apply_forces_to_boxes(boxes: &mut [BoxBody], forces: &[[f32; 2]], dt: f32) {
    for (i, box_body) in boxes.iter_mut().enumerate() {
        if i < forces.len() {
            let force = Vec3::new(forces[i][0], 0.0, forces[i][1]);
            let acceleration = force / box_body.mass;
            box_body.vel += acceleration * dt;
        }
    }
}