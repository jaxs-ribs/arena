//! # Collision Detection and Response
//! 
//! This module handles collision detection between different primitive types
//! and computes collision response using impulse-based methods.

mod sphere_sphere;
mod sphere_plane;
mod sphere_box;
mod sphere_cylinder;
mod box_plane;
mod cylinder_plane;
mod broad_phase;
mod stubs;

pub use sphere_sphere::*;
pub use sphere_plane::*;
pub use sphere_box::*;
pub use sphere_cylinder::*;
pub use box_plane::*;
pub use cylinder_plane::*;
pub use broad_phase::*;
// pub use stubs::*; // Don't re-export to avoid ambiguity

use crate::types::{Vec3, Material};

/// Contact information for collision response
#[derive(Debug, Clone, Copy)]
pub struct Contact {
    /// Contact point in world space
    pub point: Vec3,
    /// Contact normal (from body A to body B)
    pub normal: Vec3,
    /// Penetration depth (negative means separation)
    pub depth: f32,
    /// Combined friction coefficient
    pub friction: f32,
    /// Combined restitution coefficient
    pub restitution: f32,
}

impl Contact {
    /// Create a new contact with material properties
    pub fn new(point: Vec3, normal: Vec3, depth: f32, mat_a: &Material, mat_b: &Material) -> Self {
        Self {
            point,
            normal,
            depth,
            friction: combine_friction(mat_a.friction, mat_b.friction),
            restitution: combine_restitution(mat_a.restitution, mat_b.restitution),
        }
    }
}

/// Combine friction coefficients using geometric mean
fn combine_friction(f1: f32, f2: f32) -> f32 {
    (f1 * f2).sqrt()
}

/// Combine restitution coefficients using geometric mean
fn combine_restitution(r1: f32, r2: f32) -> f32 {
    (r1 * r2).sqrt()
}

/// Collision detection configuration
pub struct CollisionConfig {
    /// Contact offset for continuous collision detection
    pub contact_offset: f32,
    /// Maximum penetration before position correction
    pub max_penetration: f32,
    /// Position correction factor (0-1)
    pub position_correction: f32,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            contact_offset: 0.01,
            max_penetration: 0.1,
            position_correction: 0.8,
        }
    }
}