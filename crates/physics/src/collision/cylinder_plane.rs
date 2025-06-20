//! Cylinder-Plane collision detection and response

use crate::types::{Vec3, Cylinder, Plane};
use super::Contact;

/// Detect collision between a cylinder and a plane
pub fn detect_cylinder_plane_collision(
    cylinder: &Cylinder,
    plane: &Plane,
) -> Option<Contact> {
    // For a vertical cylinder, we check the bottom circle against the plane
    // Get the bottom center of the cylinder
    let bottom_center = cylinder.pos - Vec3::new(0.0, cylinder.half_height, 0.0);
    
    // Distance from bottom center to plane
    let center_distance = plane.normal.dot(bottom_center) + plane.d;
    
    // The closest point on the cylinder to the plane is offset by the radius
    // in the direction opposite to the plane normal (if cylinder is vertical)
    let closest_distance = center_distance - cylinder.radius * plane.normal.y.abs();
    
    // If distance is positive, no collision
    if closest_distance > 0.0 {
        return None;
    }
    
    // Contact point is the projection of the closest point onto the plane
    let contact_point = bottom_center - plane.normal * center_distance;
    
    Some(Contact::new(
        contact_point,
        plane.normal,
        -closest_distance, // Penetration depth
        &cylinder.material,
        &plane.material,
    ))
}

/// Resolve collision between a cylinder and a static plane
pub fn resolve_cylinder_plane_collision(
    cylinder: &mut Cylinder,
    plane: &Plane,
    contact: &Contact,
    dt: f32,
) {
    // Compute relative velocity at contact point
    let relative_velocity = cylinder.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Compute impulse magnitude
    let impulse_magnitude = -(1.0 + contact.restitution) * velocity_along_normal;
    
    // Apply impulse to change velocity
    let impulse = contact.normal * impulse_magnitude;
    cylinder.vel += impulse / cylinder.mass;
    
    // Position correction to resolve penetration
    if contact.depth > 0.001 {
        let correction_magnitude = contact.depth * 0.8; // 80% correction
        let correction = contact.normal * correction_magnitude;
        cylinder.pos += correction;
    }
    
    // Apply friction
    if contact.friction > 0.0 && velocity_along_normal < -0.01 {
        let tangent_velocity = relative_velocity - contact.normal * velocity_along_normal;
        let tangent_speed = tangent_velocity.length();
        
        if tangent_speed > 0.001 {
            let tangent_direction = tangent_velocity / tangent_speed;
            let friction_impulse_magnitude = contact.friction * impulse_magnitude.abs();
            let max_friction_impulse = tangent_speed * cylinder.mass / dt;
            
            let actual_friction_impulse = friction_impulse_magnitude.min(max_friction_impulse);
            cylinder.vel -= tangent_direction * (actual_friction_impulse / cylinder.mass);
        }
    }
    
    // Damp angular velocity slightly on ground contact
    cylinder.angular_vel *= 0.98;
}