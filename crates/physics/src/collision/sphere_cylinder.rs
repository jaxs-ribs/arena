//! Sphere-cylinder collision detection and response

use crate::types::{Vec3, Sphere, Cylinder};
use super::Contact;

/// Detect collision between a sphere and a cylinder
/// Note: This assumes the cylinder is axis-aligned along Y axis
pub fn detect_sphere_cylinder_collision(
    sphere: &Sphere,
    cylinder: &Cylinder,
) -> Option<Contact> {
    // Project to XZ plane for radial check
    let sphere_xz = Vec3::new(sphere.pos.x, 0.0, sphere.pos.z);
    let cylinder_xz = Vec3::new(cylinder.pos.x, 0.0, cylinder.pos.z);
    let delta_xz = sphere_xz - cylinder_xz;
    let distance_xz = delta_xz.length();
    
    // Check height bounds
    let y_min = cylinder.pos.y - cylinder.half_height;
    let y_max = cylinder.pos.y + cylinder.half_height;
    let sphere_y_clamped = sphere.pos.y.clamp(y_min, y_max);
    
    // Find closest point on cylinder
    let closest = if distance_xz > cylinder.radius {
        // Outside cylinder radius - closest point is on the edge
        let radial_dir = if distance_xz > 0.0001 {
            delta_xz / distance_xz
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        
        Vec3::new(
            cylinder.pos.x + radial_dir.x * cylinder.radius,
            sphere_y_clamped,
            cylinder.pos.z + radial_dir.z * cylinder.radius,
        )
    } else {
        // Inside cylinder radius - check end caps
        if sphere.pos.y < y_min {
            // Below cylinder
            Vec3::new(sphere.pos.x, y_min, sphere.pos.z)
        } else if sphere.pos.y > y_max {
            // Above cylinder
            Vec3::new(sphere.pos.x, y_max, sphere.pos.z)
        } else {
            // Inside cylinder - closest point is on the curved surface
            let radial_dir = if distance_xz > 0.0001 {
                delta_xz / distance_xz
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            
            Vec3::new(
                cylinder.pos.x + radial_dir.x * cylinder.radius,
                sphere.pos.y,
                cylinder.pos.z + radial_dir.z * cylinder.radius,
            )
        }
    };
    
    // Check collision
    let to_sphere = sphere.pos - closest;
    let distance = to_sphere.length();
    
    if distance < sphere.radius {
        let normal = if distance > 0.0001 {
            to_sphere / distance
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        
        let depth = sphere.radius - distance;
        let contact_point = sphere.pos - normal * sphere.radius;
        
        Some(Contact::new(
            contact_point,
            normal,
            depth,
            &sphere.material,
            &cylinder.material,
        ))
    } else {
        None
    }
}

/// Apply collision response for sphere-cylinder collision
pub fn resolve_sphere_cylinder_collision(
    sphere: &mut Sphere,
    cylinder: &mut Cylinder,
    contact: &Contact,
) {
    // Calculate relative velocity
    let relative_velocity = sphere.vel - cylinder.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Calculate impulse magnitude
    let e = contact.restitution;
    let inv_mass_sum = 1.0 / sphere.mass + 1.0 / cylinder.mass;
    let j = -(1.0 + e) * velocity_along_normal / inv_mass_sum;
    
    // Apply impulse
    let impulse = contact.normal * j;
    sphere.vel += impulse / sphere.mass;
    cylinder.vel -= impulse / cylinder.mass;
    
    // Position correction
    const POSITION_CORRECTION_PERCENT: f32 = 0.8;
    const POSITION_CORRECTION_SLOP: f32 = 0.01;
    
    let correction_magnitude = (contact.depth - POSITION_CORRECTION_SLOP).max(0.0)
        / inv_mass_sum * POSITION_CORRECTION_PERCENT;
    let correction = contact.normal * correction_magnitude;
    
    sphere.pos += correction / sphere.mass;
    cylinder.pos -= correction / cylinder.mass;
}