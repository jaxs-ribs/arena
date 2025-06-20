//! Unified primitive trait and collision detection framework

use crate::types::{Vec3, Material, Sphere, BoxBody, Cylinder, Plane};

/// Primitive shape types for collision detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    Sphere,
    Box,
    Cylinder,
    Plane,
}

/// Unified interface for collision primitives
pub trait Collider {
    /// Get the primitive type for dispatch
    fn primitive_type(&self) -> PrimitiveType;
    
    /// Get the center/position of the primitive
    fn center(&self) -> Vec3;
    
    /// Get the material properties
    fn material(&self) -> &Material;
    
    /// Get support point in given direction (for GJK algorithm)
    fn support(&self, direction: Vec3) -> Vec3;
    
    /// Simple bounding sphere radius for broad phase
    fn bounding_radius(&self) -> f32;
}

// Implement Collider for each primitive type

impl Collider for Sphere {
    fn primitive_type(&self) -> PrimitiveType {
        PrimitiveType::Sphere
    }
    
    fn center(&self) -> Vec3 {
        self.pos
    }
    
    fn material(&self) -> &Material {
        &self.material
    }
    
    fn support(&self, direction: Vec3) -> Vec3 {
        self.pos + direction.normalize() * self.radius
    }
    
    fn bounding_radius(&self) -> f32 {
        self.radius
    }
}

impl Collider for BoxBody {
    fn primitive_type(&self) -> PrimitiveType {
        PrimitiveType::Box
    }
    
    fn center(&self) -> Vec3 {
        self.pos
    }
    
    fn material(&self) -> &Material {
        &self.material
    }
    
    fn support(&self, direction: Vec3) -> Vec3 {
        // Support point for box is the corner in the given direction
        Vec3::new(
            self.pos.x + self.half_extents.x * direction.x.signum(),
            self.pos.y + self.half_extents.y * direction.y.signum(),
            self.pos.z + self.half_extents.z * direction.z.signum(),
        )
    }
    
    fn bounding_radius(&self) -> f32 {
        self.half_extents.length()
    }
}

impl Collider for Cylinder {
    fn primitive_type(&self) -> PrimitiveType {
        PrimitiveType::Cylinder
    }
    
    fn center(&self) -> Vec3 {
        self.pos
    }
    
    fn material(&self) -> &Material {
        &self.material
    }
    
    fn support(&self, direction: Vec3) -> Vec3 {
        // For a Y-axis aligned cylinder
        let dir_xz = Vec3::new(direction.x, 0.0, direction.z);
        let xz_length = dir_xz.length();
        
        // Handle degenerate case
        if xz_length < 0.0001 {
            // Pure vertical direction
            return Vec3::new(
                self.pos.x,
                self.pos.y + self.half_height * direction.y.signum(),
                self.pos.z,
            );
        }
        
        // Normalize horizontal direction
        let dir_xz_norm = dir_xz / xz_length;
        
        // Support point combines:
        // - Circle support in XZ plane
        // - Box support in Y direction
        Vec3::new(
            self.pos.x + dir_xz_norm.x * self.radius,
            self.pos.y + self.half_height * direction.y.signum(),
            self.pos.z + dir_xz_norm.z * self.radius,
        )
    }
    
    fn bounding_radius(&self) -> f32 {
        (self.radius * self.radius + self.half_height * self.half_height).sqrt()
    }
}

impl Collider for Plane {
    fn primitive_type(&self) -> PrimitiveType {
        PrimitiveType::Plane
    }
    
    fn center(&self) -> Vec3 {
        // Planes don't have a well-defined center, return origin projected onto plane
        self.normal * (-self.d)
    }
    
    fn material(&self) -> &Material {
        &self.material
    }
    
    fn support(&self, direction: Vec3) -> Vec3 {
        // Planes have infinite support in any direction not aligned with normal
        if direction.dot(self.normal) > 0.0 {
            // Can't support in normal direction
            Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY)
        } else {
            // Infinite support perpendicular to normal
            Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY)
        }
    }
    
    fn bounding_radius(&self) -> f32 {
        f32::INFINITY
    }
}

/// Dynamic primitive wrapper for polymorphic collision detection
/// Note: Using separate enums for immutable (detection) and mutable (response) phases
pub enum Primitive<'a> {
    Sphere(&'a Sphere),
    Box(&'a BoxBody),
    Cylinder(&'a Cylinder),
    Plane(&'a Plane),
}

pub enum PrimitiveMut<'a> {
    Sphere(&'a mut Sphere),
    Box(&'a mut BoxBody),
    Cylinder(&'a mut Cylinder),
    Plane(&'a mut Plane),
}

impl<'a> Primitive<'a> {
    pub fn as_collider(&self) -> &dyn Collider {
        match self {
            Primitive::Sphere(s) => *s,
            Primitive::Box(b) => *b,
            Primitive::Cylinder(c) => *c,
            Primitive::Plane(p) => *p,
        }
    }
}