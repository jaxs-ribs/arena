//! Collision detection dispatcher that routes to appropriate algorithms

use std::collections::HashMap;
use crate::types::{Vec3, Sphere, BoxBody, Cylinder, Plane};
use super::{Contact, PrimitiveType, Collider, Primitive};

// Import existing optimized collision functions
use super::{
    detect_sphere_sphere_collision,
    detect_sphere_plane_collision,
    detect_sphere_box_collision,
    detect_sphere_cylinder_collision,
    detect_box_plane_collision,
    detect_cylinder_plane_collision,
};

/// Function type for collision detection between two primitives
type CollisionDetector = fn(&dyn Collider, &dyn Collider) -> Option<Contact>;

/// Collision detection dispatcher that routes to appropriate algorithms
pub struct CollisionDispatcher {
    /// Matrix of collision detection functions indexed by primitive type pairs
    detectors: HashMap<(PrimitiveType, PrimitiveType), CollisionDetector>,
}

impl CollisionDispatcher {
    /// Create a new collision dispatcher with all detection algorithms registered
    pub fn new() -> Self {
        let mut dispatcher = Self {
            detectors: HashMap::new(),
        };
        
        // Register all collision detection functions
        dispatcher.register_detectors();
        
        dispatcher
    }
    
    /// Register all collision detection algorithms
    fn register_detectors(&mut self) {
        use PrimitiveType::*;
        
        // Sphere-Sphere
        self.register(Sphere, Sphere, detect_sphere_sphere_adapter);
        
        // Sphere-Plane
        self.register(Sphere, Plane, detect_sphere_plane_adapter);
        
        // Sphere-Box
        self.register(Sphere, Box, detect_sphere_box_adapter);
        
        // Sphere-Cylinder
        self.register(Sphere, Cylinder, detect_sphere_cylinder_adapter);
        
        // Box-Plane
        self.register(Box, Plane, detect_box_plane_adapter);
        
        // Cylinder-Plane
        self.register(Cylinder, Plane, detect_cylinder_plane_adapter);
        
        // TODO: Add more collision pairs as needed
        // For now, missing pairs will return None
    }
    
    /// Register a collision detection function for a pair of primitive types
    fn register(&mut self, type_a: PrimitiveType, type_b: PrimitiveType, detector: CollisionDetector) {
        // Register both orderings for symmetric collisions
        self.detectors.insert((type_a, type_b), detector);
        if type_a != type_b {
            self.detectors.insert((type_b, type_a), detector);
        }
    }
    
    /// Detect collision between two primitives
    pub fn detect(&self, prim_a: &Primitive, prim_b: &Primitive) -> Option<Contact> {
        let collider_a = prim_a.as_collider();
        let collider_b = prim_b.as_collider();
        
        let key = (collider_a.primitive_type(), collider_b.primitive_type());
        
        if let Some(detector) = self.detectors.get(&key) {
            detector(collider_a, collider_b)
        } else {
            // No detector registered for this pair
            None
        }
    }
}

// Adapter functions to convert between trait objects and concrete types

fn detect_sphere_sphere_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    // SAFETY: We know these are spheres from the dispatcher
    let sphere_a = unsafe { &*(a as *const dyn Collider as *const Sphere) };
    let sphere_b = unsafe { &*(b as *const dyn Collider as *const Sphere) };
    detect_sphere_sphere_collision(sphere_a, sphere_b)
}

fn detect_sphere_plane_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    // Handle both orderings
    if a.primitive_type() == PrimitiveType::Sphere {
        let sphere = unsafe { &*(a as *const dyn Collider as *const Sphere) };
        let plane = unsafe { &*(b as *const dyn Collider as *const Plane) };
        detect_sphere_plane_collision(sphere, plane)
    } else {
        let sphere = unsafe { &*(b as *const dyn Collider as *const Sphere) };
        let plane = unsafe { &*(a as *const dyn Collider as *const Plane) };
        detect_sphere_plane_collision(sphere, plane)
    }
}

fn detect_sphere_box_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    if a.primitive_type() == PrimitiveType::Sphere {
        let sphere = unsafe { &*(a as *const dyn Collider as *const Sphere) };
        let box_body = unsafe { &*(b as *const dyn Collider as *const BoxBody) };
        detect_sphere_box_collision(sphere, box_body)
    } else {
        let sphere = unsafe { &*(b as *const dyn Collider as *const Sphere) };
        let box_body = unsafe { &*(a as *const dyn Collider as *const BoxBody) };
        detect_sphere_box_collision(sphere, box_body)
    }
}

fn detect_sphere_cylinder_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    if a.primitive_type() == PrimitiveType::Sphere {
        let sphere = unsafe { &*(a as *const dyn Collider as *const Sphere) };
        let cylinder = unsafe { &*(b as *const dyn Collider as *const Cylinder) };
        detect_sphere_cylinder_collision(sphere, cylinder)
    } else {
        let sphere = unsafe { &*(b as *const dyn Collider as *const Sphere) };
        let cylinder = unsafe { &*(a as *const dyn Collider as *const Cylinder) };
        detect_sphere_cylinder_collision(sphere, cylinder)
    }
}

fn detect_box_plane_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    if a.primitive_type() == PrimitiveType::Box {
        let box_body = unsafe { &*(a as *const dyn Collider as *const BoxBody) };
        let plane = unsafe { &*(b as *const dyn Collider as *const Plane) };
        detect_box_plane_collision(box_body, plane)
    } else {
        let box_body = unsafe { &*(b as *const dyn Collider as *const BoxBody) };
        let plane = unsafe { &*(a as *const dyn Collider as *const Plane) };
        detect_box_plane_collision(box_body, plane)
    }
}

fn detect_cylinder_plane_adapter(a: &dyn Collider, b: &dyn Collider) -> Option<Contact> {
    if a.primitive_type() == PrimitiveType::Cylinder {
        let cylinder = unsafe { &*(a as *const dyn Collider as *const Cylinder) };
        let plane = unsafe { &*(b as *const dyn Collider as *const Plane) };
        detect_cylinder_plane_collision(cylinder, plane)
    } else {
        let cylinder = unsafe { &*(b as *const dyn Collider as *const Cylinder) };
        let plane = unsafe { &*(a as *const dyn Collider as *const Plane) };
        detect_cylinder_plane_collision(cylinder, plane)
    }
}

impl Default for CollisionDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Vec3, Material};
    
    #[test]
    fn test_dispatcher_sphere_sphere() {
        let dispatcher = CollisionDispatcher::new();
        
        let sphere_a = Sphere::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        let sphere_b = Sphere::new(Vec3::new(1.5, 0.0, 0.0), Vec3::ZERO, 1.0);
        
        let prim_a = Primitive::Sphere(&sphere_a);
        let prim_b = Primitive::Sphere(&sphere_b);
        
        let contact = dispatcher.detect(&prim_a, &prim_b);
        assert!(contact.is_some());
    }
}