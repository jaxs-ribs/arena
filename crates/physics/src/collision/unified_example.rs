//! Example of using the unified collision system
//! This shows how the new architecture simplifies collision handling

use crate::types::{Vec3, Sphere, BoxBody, Cylinder, Plane};
use crate::collision::{CollisionDispatcher, CollisionSolver, Primitive};

/// Example of the new unified collision detection and response
pub fn unified_collision_example(
    spheres: &mut Vec<Sphere>,
    boxes: &mut Vec<BoxBody>,
    cylinders: &mut Vec<Cylinder>,
    planes: &Vec<Plane>,
    dt: f32,
) {
    // Create dispatcher and solver once
    let dispatcher = CollisionDispatcher::new();
    let solver = CollisionSolver::new();
    
    // Check all sphere-sphere collisions
    for i in 0..spheres.len() {
        for j in (i + 1)..spheres.len() {
            let (left, right) = spheres.split_at_mut(j);
            let sphere_a = &mut left[i];
            let sphere_b = &mut right[0];
            
            let mut prim_a = Primitive::Sphere(sphere_a);
            let mut prim_b = Primitive::Sphere(sphere_b);
            
            if let Some(contact) = dispatcher.detect(&prim_a, &prim_b) {
                solver.resolve(&mut prim_a, &mut prim_b, &contact, dt);
            }
        }
    }
    
    // Check all dynamic vs static collisions
    // Note: This is just an example - actual implementation would need
    // to handle mutable/immutable reference splitting properly
    
    for box_body in boxes.iter_mut() {
        for plane in planes.iter() {
            let mut prim_box = Primitive::Box(box_body);
            let mut prim_plane = Primitive::Plane(plane);
            
            if let Some(contact) = dispatcher.detect(&prim_box, &prim_plane) {
                solver.resolve(&mut prim_box, &mut prim_plane, &contact, dt);
            }
        }
    }
    
    for cylinder in cylinders.iter_mut() {
        for plane in planes.iter() {
            let mut prim_cylinder = Primitive::Cylinder(cylinder);
            let mut prim_plane = Primitive::Plane(plane);
            
            if let Some(contact) = dispatcher.detect(&prim_cylinder, &prim_plane) {
                solver.resolve(&mut prim_cylinder, &mut prim_plane, &contact, dt);
            }
        }
    }
}

/// Future: Even cleaner with a unified primitive collection
pub struct UnifiedSimulation {
    dispatcher: CollisionDispatcher,
    solver: CollisionSolver,
}

impl UnifiedSimulation {
    pub fn new() -> Self {
        Self {
            dispatcher: CollisionDispatcher::new(),
            solver: CollisionSolver::new(),
        }
    }
    
    /// Check collision between any two primitives
    pub fn check_and_resolve(&self, prim_a: &mut Primitive, prim_b: &mut Primitive, dt: f32) {
        if let Some(contact) = self.dispatcher.detect(prim_a, prim_b) {
            self.solver.resolve(prim_a, prim_b, &contact, dt);
        }
    }
}