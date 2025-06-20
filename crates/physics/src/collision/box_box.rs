//! Box-box collision detection and response

use crate::types::{Vec3, BoxBody, Material};
use super::{Contact, combine_friction, combine_restitution};

/// Detect collision between two boxes
pub fn detect_box_box_collision(
    box_a: &BoxBody,
    box_b: &BoxBody,
) -> Option<Contact> {
    // Calculate separation on each axis
    let center_diff = box_b.pos - box_a.pos;
    let combined_half_extents = box_a.half_extents + box_b.half_extents;
    
    // Check if boxes overlap on all axes
    let overlap_x = combined_half_extents.x - center_diff.x.abs();
    let overlap_y = combined_half_extents.y - center_diff.y.abs();
    let overlap_z = combined_half_extents.z - center_diff.z.abs();
    
    if overlap_x > 0.0 && overlap_y > 0.0 && overlap_z > 0.0 {
        // Find axis with smallest overlap (separation axis)
        let min_overlap = overlap_x.min(overlap_y).min(overlap_z);
        
        let normal = if overlap_x == min_overlap {
            Vec3::new(center_diff.x.signum(), 0.0, 0.0)
        } else if overlap_y == min_overlap {
            Vec3::new(0.0, center_diff.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, center_diff.z.signum())
        };
        
        // Contact point is at the midpoint between the boxes along the normal
        let contact_point = box_a.pos + normal * (box_a.half_extents.dot(normal.abs()));
        
        Some(Contact::new(
            contact_point,
            normal,
            min_overlap,
            &box_a.material,
            &box_b.material,
        ))
    } else {
        None
    }
}

/// Resolve collision between two boxes
pub fn resolve_box_box_collision(
    box_a: &mut BoxBody,
    box_b: &mut BoxBody,
    contact: &Contact,
) {
    // Calculate relative velocity
    let relative_velocity = box_b.vel - box_a.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Calculate impulse
    let mass_sum = 1.0 / box_a.mass + 1.0 / box_b.mass;
    let restitution = contact.restitution;
    let impulse_magnitude = -(1.0 + restitution) * velocity_along_normal / mass_sum;
    let impulse = contact.normal * impulse_magnitude;
    
    // Apply impulse
    box_a.vel -= impulse / box_a.mass;
    box_b.vel += impulse / box_b.mass;
    
    // Position correction to resolve penetration
    if contact.depth > 0.001 {
        let correction = contact.normal * contact.depth * 0.8;
        let mass_ratio_a = (1.0 / box_a.mass) / mass_sum;
        let mass_ratio_b = (1.0 / box_b.mass) / mass_sum;
        
        box_a.pos -= correction * mass_ratio_a;
        box_b.pos += correction * mass_ratio_b;
    }
    
    // Apply friction
    apply_friction(box_a, box_b, contact, impulse_magnitude);
}

fn apply_friction(
    box_a: &mut BoxBody,
    box_b: &mut BoxBody,
    contact: &Contact,
    normal_impulse: f32,
) {
    if contact.friction <= 0.0 || normal_impulse <= 0.0 {
        return;
    }
    
    let relative_velocity = box_b.vel - box_a.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    let tangent_velocity = relative_velocity - contact.normal * velocity_along_normal;
    let tangent_speed = tangent_velocity.length();
    
    if tangent_speed < 0.001 {
        return;
    }
    
    let tangent_direction = tangent_velocity / tangent_speed;
    let mass_sum = 1.0 / box_a.mass + 1.0 / box_b.mass;
    let max_friction = tangent_speed / mass_sum;
    let friction_impulse = (contact.friction * normal_impulse.abs()).min(max_friction);
    
    let friction = tangent_direction * friction_impulse;
    box_a.vel += friction / box_a.mass;
    box_b.vel -= friction / box_b.mass;
}