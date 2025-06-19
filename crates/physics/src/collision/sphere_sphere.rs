//! Sphere-sphere collision detection and response

use crate::types::{Vec3, Sphere};
use super::Contact;

/// Detect collision between two spheres
pub fn detect_sphere_sphere_collision(
    sphere_a: &Sphere,
    sphere_b: &Sphere,
) -> Option<Contact> {
    let delta = sphere_b.pos - sphere_a.pos;
    let distance_squared = delta.dot(delta);
    let min_distance = sphere_a.radius + sphere_b.radius;
    
    if distance_squared < min_distance * min_distance {
        let distance = distance_squared.sqrt();
        
        // Handle case where spheres are at same position
        let normal = if distance > 0.0001 {
            delta / distance
        } else {
            Vec3::new(0.0, 1.0, 0.0) // Default up direction
        };
        
        let depth = min_distance - distance;
        let contact_point = sphere_a.pos + normal * sphere_a.radius;
        
        Some(Contact::new(
            contact_point,
            normal,
            depth,
            &sphere_a.material,
            &sphere_b.material,
        ))
    } else {
        None
    }
}

/// Apply impulse-based collision response between two spheres
pub fn resolve_sphere_sphere_collision(
    sphere_a: &mut Sphere,
    sphere_b: &mut Sphere,
    contact: &Contact,
) {
    // Calculate relative velocity
    let relative_velocity = sphere_b.vel - sphere_a.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Calculate impulse magnitude
    let e = contact.restitution;
    let inv_mass_sum = 1.0 / sphere_a.mass + 1.0 / sphere_b.mass;
    let j = -(1.0 + e) * velocity_along_normal / inv_mass_sum;
    
    // Apply impulse
    let impulse = contact.normal * j;
    sphere_a.vel -= impulse / sphere_a.mass;
    sphere_b.vel += impulse / sphere_b.mass;
    
    // Position correction to resolve penetration
    const POSITION_CORRECTION_PERCENT: f32 = 0.8;
    const POSITION_CORRECTION_SLOP: f32 = 0.01;
    
    let correction_magnitude = (contact.depth - POSITION_CORRECTION_SLOP).max(0.0)
        / inv_mass_sum * POSITION_CORRECTION_PERCENT;
    let correction = contact.normal * correction_magnitude;
    
    sphere_a.pos -= correction / sphere_a.mass;
    sphere_b.pos += correction / sphere_b.mass;
}