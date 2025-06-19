//! Sphere-plane collision detection and response

use crate::types::{Sphere, Plane};
use super::Contact;

/// Detect collision between a sphere and a plane
pub fn detect_sphere_plane_collision(
    sphere: &Sphere,
    plane: &Plane,
) -> Option<Contact> {
    // Distance from sphere center to plane
    let distance = sphere.pos.dot(plane.normal) - plane.d;
    
    // Check if sphere intersects plane
    if distance < sphere.radius {
        // Contact point is the closest point on the sphere to the plane
        let contact_point = sphere.pos - plane.normal * sphere.radius;
        let depth = sphere.radius - distance;
        
        Some(Contact::new(
            contact_point,
            plane.normal,
            depth,
            &sphere.material,
            &plane.material,
        ))
    } else {
        None
    }
}

/// Apply collision response for sphere-plane collision
pub fn resolve_sphere_plane_collision(
    sphere: &mut Sphere,
    plane: &Plane,
    contact: &Contact,
) {
    // Calculate velocity along normal
    let velocity_along_normal = sphere.vel.dot(contact.normal);
    
    // Only resolve if moving towards plane
    if velocity_along_normal < 0.0 {
        // Calculate impulse with restitution
        let e = contact.restitution;
        let j = -(1.0 + e) * velocity_along_normal;
        
        // Apply impulse to velocity
        sphere.vel += contact.normal * j;
        
        // Apply friction
        apply_friction(sphere, contact, velocity_along_normal);
    }
    
    // Position correction to prevent sinking
    if contact.depth > 0.01 {
        sphere.pos += contact.normal * (contact.depth * 0.8);
    }
}

/// Apply friction impulse to sphere
fn apply_friction(sphere: &mut Sphere, contact: &Contact, normal_impulse: f32) {
    // Calculate tangent velocity
    let velocity_along_normal = sphere.vel.dot(contact.normal);
    let normal_velocity = contact.normal * velocity_along_normal;
    let tangent_velocity = sphere.vel - normal_velocity;
    
    let tangent_speed = tangent_velocity.length();
    if tangent_speed > 0.0001 {
        let tangent_direction = tangent_velocity / tangent_speed;
        
        // Friction impulse magnitude
        let friction_impulse_magnitude = contact.friction * normal_impulse.abs();
        
        // Apply friction (clamped to not reverse motion)
        let friction_impulse = tangent_direction * 
            friction_impulse_magnitude.min(tangent_speed);
        
        sphere.vel -= friction_impulse;
    }
}