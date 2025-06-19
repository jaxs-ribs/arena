//! Sphere-box collision detection and response

use crate::types::{Vec3, Sphere, BoxBody};
use super::Contact;

/// Detect collision between a sphere and an axis-aligned box
pub fn detect_sphere_box_collision(
    sphere: &Sphere,
    box_body: &BoxBody,
) -> Option<Contact> {
    // Find closest point on box to sphere center
    let closest = closest_point_on_box(sphere.pos, box_body);
    
    // Check if closest point is within sphere radius
    let delta = closest - sphere.pos;
    let distance_squared = delta.dot(delta);
    
    if distance_squared < sphere.radius * sphere.radius {
        let distance = distance_squared.sqrt();
        
        // Calculate contact normal
        let normal = if distance > 0.0001 {
            (sphere.pos - closest) / distance
        } else {
            // Sphere center is inside box, find closest face
            find_closest_face_normal(sphere.pos, box_body)
        };
        
        let depth = sphere.radius - distance;
        let contact_point = sphere.pos - normal * sphere.radius;
        
        Some(Contact::new(
            contact_point,
            normal,
            depth,
            &sphere.material,
            &box_body.material,
        ))
    } else {
        None
    }
}

/// Find the closest point on an axis-aligned box to a given point
fn closest_point_on_box(point: Vec3, box_body: &BoxBody) -> Vec3 {
    let min = box_body.pos - box_body.half_extents;
    let max = box_body.pos + box_body.half_extents;
    
    Vec3::new(
        point.x.clamp(min.x, max.x),
        point.y.clamp(min.y, max.y),
        point.z.clamp(min.z, max.z),
    )
}

/// Find the normal of the closest box face to a point inside the box
fn find_closest_face_normal(point: Vec3, box_body: &BoxBody) -> Vec3 {
    let local_point = point - box_body.pos;
    let abs_local = Vec3::new(
        local_point.x.abs(),
        local_point.y.abs(),
        local_point.z.abs(),
    );
    
    let distances = Vec3::new(
        box_body.half_extents.x - abs_local.x,
        box_body.half_extents.y - abs_local.y,
        box_body.half_extents.z - abs_local.z,
    );
    
    // Find the axis with minimum distance to face
    if distances.x < distances.y && distances.x < distances.z {
        Vec3::new(local_point.x.signum(), 0.0, 0.0)
    } else if distances.y < distances.z {
        Vec3::new(0.0, local_point.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, local_point.z.signum())
    }
}

/// Apply collision response for sphere-box collision
pub fn resolve_sphere_box_collision(
    sphere: &mut Sphere,
    box_body: &mut BoxBody,
    contact: &Contact,
) {
    // Calculate relative velocity
    let relative_velocity = sphere.vel - box_body.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Calculate impulse magnitude
    let e = contact.restitution;
    let inv_mass_sum = 1.0 / sphere.mass + 1.0 / box_body.mass;
    let j = -(1.0 + e) * velocity_along_normal / inv_mass_sum;
    
    // Apply impulse
    let impulse = contact.normal * j;
    sphere.vel += impulse / sphere.mass;
    box_body.vel -= impulse / box_body.mass;
    
    // Position correction
    const POSITION_CORRECTION_PERCENT: f32 = 0.8;
    const POSITION_CORRECTION_SLOP: f32 = 0.01;
    
    let correction_magnitude = (contact.depth - POSITION_CORRECTION_SLOP).max(0.0)
        / inv_mass_sum * POSITION_CORRECTION_PERCENT;
    let correction = contact.normal * correction_magnitude;
    
    sphere.pos += correction / sphere.mass;
    box_body.pos -= correction / box_body.mass;
}