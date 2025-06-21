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
    use crate::types::BodyType;
    
    for box_body in boxes.iter_mut() {
        // Only apply gravity to dynamic bodies
        if box_body.body_type == BodyType::Dynamic {
            let acceleration = gravity;
            box_body.vel += acceleration * dt;
        }
        
        // Update position for dynamic and kinematic bodies (static bodies don't move)
        if box_body.body_type != BodyType::Static {
            box_body.pos += box_body.vel * dt;
        }
        
        // Integrate angular velocity
        if box_body.angular_vel.length() > 0.0 {
            let angle = box_body.angular_vel.length() * dt;
            let axis = box_body.angular_vel.normalize();
            let delta_quat = quaternion_from_axis_angle(axis, angle);
            box_body.orientation = quaternion_multiply(delta_quat, box_body.orientation);
            normalize_quaternion(&mut box_body.orientation);
        }
        
        // Apply damping (disabled for now)
        // box_body.vel *= DAMPING_FACTOR;
        // box_body.angular_vel *= DAMPING_FACTOR;
    }
}

/// Integrate cylinder positions and velocities
pub fn integrate_cylinders(cylinders: &mut [Cylinder], gravity: Vec3, dt: f32) {
    use crate::types::BodyType;
    
    for cylinder in cylinders.iter_mut() {
        // SKIP gravity for dynamic cylinders - let the constraint solver handle forces
        // This prevents joint drift in CartPole systems
        // (Direct gravity application conflicts with rigid joint constraints)
        
        // Update position for dynamic and kinematic bodies (static bodies don't move)
        if cylinder.body_type != BodyType::Static {
            cylinder.pos += cylinder.vel * dt;
        }
        
        // Integrate angular velocity and update orientation
        if cylinder.angular_vel.length() > 0.0 {
            let angle = cylinder.angular_vel.length() * dt;
            let axis = cylinder.angular_vel.normalize();
            let delta_quat = quaternion_from_axis_angle(axis, angle);
            cylinder.orientation = quaternion_multiply(delta_quat, cylinder.orientation);
            normalize_quaternion(&mut cylinder.orientation);
        }
        
        // Apply damping (disabled for now)  
        // cylinder.vel *= DAMPING_FACTOR;
        // cylinder.angular_vel *= DAMPING_FACTOR;
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
    use crate::types::BodyType;
    
    for (i, box_body) in boxes.iter_mut().enumerate() {
        // Only apply forces to dynamic bodies
        // Kinematic bodies are controlled by setting velocity directly
        if box_body.body_type == BodyType::Dynamic && i < forces.len() {
            let force = Vec3::new(forces[i][0], 0.0, forces[i][1]);
            let acceleration = force / box_body.mass;
            box_body.vel += acceleration * dt;
        }
    }
}

// Quaternion helper functions
fn quaternion_from_axis_angle(axis: Vec3, angle: f32) -> [f32; 4] {
    let half_angle = angle * 0.5;
    let s = half_angle.sin();
    [
        axis.x * s,
        axis.y * s,
        axis.z * s,
        half_angle.cos(),
    ]
}

fn quaternion_multiply(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    [
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    ]
}

fn normalize_quaternion(q: &mut [f32; 4]) {
    let mag = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if mag > 0.0 {
        q[0] /= mag;
        q[1] /= mag;
        q[2] /= mag;
        q[3] /= mag;
    }
}