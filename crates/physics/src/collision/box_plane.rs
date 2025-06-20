//! Box-Plane collision detection and response

use crate::types::{Vec3, BoxBody, Plane};
use super::Contact;

/// Detect collision between a box and a plane
pub fn detect_box_plane_collision(
    box_body: &BoxBody,
    plane: &Plane,
) -> Option<Contact> {
    // Find the box corner that's most negative along the plane normal
    // This is the "support point" in the negative normal direction
    let support_point = find_box_support_point(box_body, -plane.normal);
    
    // Compute signed distance from support point to plane
    let distance = plane.normal.dot(support_point) + plane.d;
    
    // If distance is positive, no collision
    if distance > 0.0 {
        return None;
    }
    
    // Contact point is the projection of the support point onto the plane
    let contact_point = support_point - plane.normal * distance;
    
    Some(Contact::new(
        contact_point,
        plane.normal,
        -distance, // Penetration depth
        &box_body.material,
        &plane.material,
    ))
}

/// Find the point on the box that is furthest in a given direction
fn find_box_support_point(box_body: &BoxBody, direction: Vec3) -> Vec3 {
    // For an axis-aligned box, the support point is at one of the 8 corners
    // We select the corner by checking the sign of the direction along each axis
    Vec3::new(
        box_body.pos.x + box_body.half_extents.x * direction.x.signum(),
        box_body.pos.y + box_body.half_extents.y * direction.y.signum(),
        box_body.pos.z + box_body.half_extents.z * direction.z.signum(),
    )
}

/// Resolve collision between a box and a static plane
pub fn resolve_box_plane_collision(
    box_body: &mut BoxBody,
    plane: &Plane,
    contact: &Contact,
    dt: f32,
) {
    // Compute relative velocity at contact point
    let relative_velocity = box_body.vel;
    let velocity_along_normal = relative_velocity.dot(contact.normal);
    
    // Don't resolve if velocities are separating
    if velocity_along_normal > 0.0 {
        return;
    }
    
    // Compute impulse magnitude
    let impulse_magnitude = -(1.0 + contact.restitution) * velocity_along_normal;
    
    // Apply impulse to change velocity
    let impulse = contact.normal * impulse_magnitude;
    box_body.vel += impulse / box_body.mass;
    
    // Position correction to resolve penetration
    if contact.depth > 0.001 {
        let correction_magnitude = contact.depth * 0.8; // 80% correction
        let correction = contact.normal * correction_magnitude;
        box_body.pos += correction;
    }
    
    // Apply friction
    if contact.friction > 0.0 && velocity_along_normal < -0.01 {
        let tangent_velocity = relative_velocity - contact.normal * velocity_along_normal;
        let tangent_speed = tangent_velocity.length();
        
        if tangent_speed > 0.001 {
            let tangent_direction = tangent_velocity / tangent_speed;
            let friction_impulse_magnitude = contact.friction * impulse_magnitude.abs();
            let max_friction_impulse = tangent_speed * box_body.mass / dt;
            
            let actual_friction_impulse = friction_impulse_magnitude.min(max_friction_impulse);
            box_body.vel -= tangent_direction * (actual_friction_impulse / box_body.mass);
        }
    }
    
    // Damp angular velocity slightly on ground contact
    box_body.angular_vel *= 0.98;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Vec2, Material};

    #[test]
    fn test_box_resting_on_plane() {
        let mut box_body = BoxBody {
            pos: Vec3::new(0.0, 0.5, 0.0),
            half_extents: Vec3::new(0.5, 0.5, 0.5),
            vel: Vec3::new(0.0, -1.0, 0.0),
            mass: 1.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::ZERO,
            material: Material::default(),
            body_type: crate::types::BodyType::Dynamic,
        };
        
        let plane = Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
            extents: Vec2::new(10.0, 10.0),
            material: Material::default(),
        };
        
        // Detect collision
        let contact = detect_box_plane_collision(&box_body, &plane);
        assert!(contact.is_some());
        
        let contact = contact.unwrap();
        assert!((contact.depth - 0.0).abs() < 0.001); // Box is just touching
        
        // Resolve collision
        resolve_box_plane_collision(&mut box_body, &plane, &contact, 0.016);
        
        // Box should have bounced
        assert!(box_body.vel.y > 0.0);
    }
}