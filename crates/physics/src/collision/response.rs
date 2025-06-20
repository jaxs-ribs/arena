//! Unified collision response system

use crate::types::{Vec3, Sphere, BoxBody, Cylinder, Plane};
use super::{Contact, PrimitiveType, PrimitiveMut};

// Import existing resolution functions
use super::{
    resolve_sphere_sphere_collision,
    resolve_sphere_plane_collision,
    resolve_sphere_box_collision,
    resolve_sphere_cylinder_collision,
    resolve_box_plane_collision,
    resolve_cylinder_plane_collision,
};

/// Trait for objects that can respond to collisions
pub trait CollisionResponder {
    /// Get current velocity
    fn velocity(&self) -> Vec3;
    
    /// Get mass (infinite for static objects)
    fn mass(&self) -> f32;
    
    /// Apply impulse to change velocity
    fn apply_impulse(&mut self, impulse: Vec3);
    
    /// Apply position correction
    fn apply_correction(&mut self, correction: Vec3);
    
    /// Get angular velocity (for rotating bodies)
    fn angular_velocity(&self) -> Vec3 {
        Vec3::ZERO
    }
    
    /// Apply angular impulse
    fn apply_angular_impulse(&mut self, _impulse: Vec3) {
        // Default: no rotation
    }
}

// Implement CollisionResponder for each type

impl CollisionResponder for Sphere {
    fn velocity(&self) -> Vec3 {
        self.vel
    }
    
    fn mass(&self) -> f32 {
        self.mass
    }
    
    fn apply_impulse(&mut self, impulse: Vec3) {
        self.vel += impulse / self.mass;
    }
    
    fn apply_correction(&mut self, correction: Vec3) {
        self.pos += correction;
    }
    
    fn angular_velocity(&self) -> Vec3 {
        self.angular_vel
    }
    
    fn apply_angular_impulse(&mut self, impulse: Vec3) {
        // Simplified: treat as point mass for now
        self.angular_vel += impulse / self.mass;
    }
}

impl CollisionResponder for BoxBody {
    fn velocity(&self) -> Vec3 {
        self.vel
    }
    
    fn mass(&self) -> f32 {
        self.mass
    }
    
    fn apply_impulse(&mut self, impulse: Vec3) {
        self.vel += impulse / self.mass;
    }
    
    fn apply_correction(&mut self, correction: Vec3) {
        self.pos += correction;
    }
    
    fn angular_velocity(&self) -> Vec3 {
        self.angular_vel
    }
    
    fn apply_angular_impulse(&mut self, impulse: Vec3) {
        self.angular_vel += impulse / self.mass;
    }
}

impl CollisionResponder for Cylinder {
    fn velocity(&self) -> Vec3 {
        self.vel
    }
    
    fn mass(&self) -> f32 {
        self.mass
    }
    
    fn apply_impulse(&mut self, impulse: Vec3) {
        self.vel += impulse / self.mass;
    }
    
    fn apply_correction(&mut self, correction: Vec3) {
        self.pos += correction;
    }
    
    fn angular_velocity(&self) -> Vec3 {
        self.angular_vel
    }
    
    fn apply_angular_impulse(&mut self, impulse: Vec3) {
        self.angular_vel += impulse / self.mass;
    }
}

impl CollisionResponder for Plane {
    fn velocity(&self) -> Vec3 {
        Vec3::ZERO // Static
    }
    
    fn mass(&self) -> f32 {
        f32::INFINITY // Infinite mass
    }
    
    fn apply_impulse(&mut self, _impulse: Vec3) {
        // Static object - no response
    }
    
    fn apply_correction(&mut self, _correction: Vec3) {
        // Static object - no correction
    }
}

/// Unified collision response solver
pub struct CollisionSolver {
    /// Position correction factor (0-1)
    pub position_correction: f32,
    /// Velocity correction factor for restitution
    pub restitution_slop: f32,
}

impl CollisionSolver {
    pub fn new() -> Self {
        Self {
            position_correction: 0.8,  // 80% correction per frame
            restitution_slop: 0.01,    // Minimum velocity for restitution
        }
    }
    
    /// Resolve collision between two primitives with contact information
    pub fn resolve(&self, prim_a: &mut PrimitiveMut, prim_b: &mut PrimitiveMut, contact: &Contact, dt: f32) {
        match (prim_a, prim_b) {
            (PrimitiveMut::Sphere(s1), PrimitiveMut::Sphere(s2)) => {
                // Need to handle mutable reference splitting
                // For now, use existing sphere-sphere resolution
                resolve_sphere_sphere_collision(s1, s2, contact);
            }
            (PrimitiveMut::Sphere(s), PrimitiveMut::Plane(p)) => {
                resolve_sphere_plane_collision(s, p, contact);
            }
            (PrimitiveMut::Box(b), PrimitiveMut::Plane(p)) => {
                resolve_box_plane_collision(b, p, contact, dt);
            }
            (PrimitiveMut::Cylinder(c), PrimitiveMut::Plane(p)) => {
                resolve_cylinder_plane_collision(c, p, contact, dt);
            }
            (PrimitiveMut::Sphere(s), PrimitiveMut::Box(b)) => {
                resolve_sphere_box_collision(s, b, contact);
            }
            (PrimitiveMut::Sphere(s), PrimitiveMut::Cylinder(c)) => {
                resolve_sphere_cylinder_collision(s, c, contact);
            }
            // Symmetric cases
            (PrimitiveMut::Plane(p), PrimitiveMut::Sphere(s)) => {
                resolve_sphere_plane_collision(s, p, contact);
            }
            (PrimitiveMut::Plane(p), PrimitiveMut::Box(b)) => {
                resolve_box_plane_collision(b, p, contact, dt);
            }
            (PrimitiveMut::Plane(p), PrimitiveMut::Cylinder(c)) => {
                resolve_cylinder_plane_collision(c, p, contact, dt);
            }
            (PrimitiveMut::Box(b), PrimitiveMut::Sphere(s)) => {
                resolve_sphere_box_collision(s, b, contact);
            }
            (PrimitiveMut::Cylinder(c), PrimitiveMut::Sphere(s)) => {
                resolve_sphere_cylinder_collision(s, c, contact);
            }
            // Add more pairs as needed
            _ => {} // Unsupported pair
        }
    }
    
    /// Resolve collision between two dynamic objects
    fn resolve_dynamic_dynamic<A: CollisionResponder, B: CollisionResponder>(
        &self,
        obj_a: &mut A,
        obj_b: &mut B,
        contact: &Contact,
        dt: f32,
    ) {
        // Calculate relative velocity
        let relative_velocity = obj_a.velocity() - obj_b.velocity();
        let velocity_along_normal = relative_velocity.dot(contact.normal);
        
        // Don't resolve if velocities are separating
        if velocity_along_normal > 0.0 {
            return;
        }
        
        // Calculate restitution (bounciness)
        let restitution = if velocity_along_normal.abs() > self.restitution_slop {
            contact.restitution
        } else {
            0.0 // No bounce at low speeds
        };
        
        // Calculate impulse scalar
        let mass_sum = 1.0 / obj_a.mass() + 1.0 / obj_b.mass();
        let impulse_magnitude = -(1.0 + restitution) * velocity_along_normal / mass_sum;
        
        // Apply impulse
        let impulse = contact.normal * impulse_magnitude;
        obj_a.apply_impulse(impulse);
        obj_b.apply_impulse(-impulse);
        
        // Position correction to resolve penetration
        if contact.depth > 0.001 {
            let total_correction = contact.normal * contact.depth * self.position_correction;
            let mass_ratio_a = (1.0 / obj_a.mass()) / mass_sum;
            let mass_ratio_b = (1.0 / obj_b.mass()) / mass_sum;
            
            obj_a.apply_correction(total_correction * mass_ratio_a);
            obj_b.apply_correction(-total_correction * mass_ratio_b);
        }
        
        // Apply friction
        self.apply_friction(obj_a, obj_b, contact, impulse_magnitude, dt);
    }
    
    /// Resolve collision between dynamic and static objects
    fn resolve_dynamic_static<D: CollisionResponder, S: CollisionResponder>(
        &self,
        dynamic: &mut D,
        static_obj: &S,
        contact: &Contact,
        dt: f32,
    ) {
        // Calculate relative velocity
        let relative_velocity = dynamic.velocity() - static_obj.velocity();
        let velocity_along_normal = relative_velocity.dot(contact.normal);
        
        // Don't resolve if velocities are separating
        if velocity_along_normal > 0.0 {
            return;
        }
        
        // Calculate restitution
        let restitution = if velocity_along_normal.abs() > self.restitution_slop {
            contact.restitution
        } else {
            0.0
        };
        
        // Calculate impulse (simplified for static object)
        let impulse_magnitude = -(1.0 + restitution) * velocity_along_normal;
        let impulse = contact.normal * impulse_magnitude;
        
        // Apply impulse only to dynamic object
        dynamic.apply_impulse(impulse);
        
        // Position correction
        if contact.depth > 0.001 {
            let correction = contact.normal * contact.depth * self.position_correction;
            dynamic.apply_correction(correction);
        }
        
        // Apply friction
        self.apply_friction_static(dynamic, contact, impulse_magnitude, dt);
    }
    
    /// Apply friction between two dynamic objects
    fn apply_friction<A: CollisionResponder, B: CollisionResponder>(
        &self,
        obj_a: &mut A,
        obj_b: &mut B,
        contact: &Contact,
        normal_impulse: f32,
        dt: f32,
    ) {
        if contact.friction <= 0.0 || normal_impulse <= 0.0 {
            return;
        }
        
        // Calculate relative velocity
        let relative_velocity = obj_a.velocity() - obj_b.velocity();
        let velocity_along_normal = relative_velocity.dot(contact.normal);
        
        // Get tangent velocity
        let tangent_velocity = relative_velocity - contact.normal * velocity_along_normal;
        let tangent_speed = tangent_velocity.length();
        
        if tangent_speed < 0.001 {
            return;
        }
        
        let tangent_direction = tangent_velocity / tangent_speed;
        
        // Calculate friction impulse
        let mass_sum = 1.0 / obj_a.mass() + 1.0 / obj_b.mass();
        let max_friction = tangent_speed / mass_sum;
        let friction_impulse = (contact.friction * normal_impulse.abs()).min(max_friction);
        
        let friction = tangent_direction * friction_impulse;
        obj_a.apply_impulse(-friction);
        obj_b.apply_impulse(friction);
    }
    
    /// Apply friction for dynamic-static collision
    fn apply_friction_static<D: CollisionResponder>(
        &self,
        dynamic: &mut D,
        contact: &Contact,
        normal_impulse: f32,
        dt: f32,
    ) {
        if contact.friction <= 0.0 || normal_impulse <= 0.0 {
            return;
        }
        
        let velocity = dynamic.velocity();
        let velocity_along_normal = velocity.dot(contact.normal);
        let tangent_velocity = velocity - contact.normal * velocity_along_normal;
        let tangent_speed = tangent_velocity.length();
        
        if tangent_speed < 0.001 {
            return;
        }
        
        let tangent_direction = tangent_velocity / tangent_speed;
        let max_friction = tangent_speed * dynamic.mass() / dt;
        let friction_impulse = (contact.friction * normal_impulse.abs()).min(max_friction);
        
        dynamic.apply_impulse(-tangent_direction * friction_impulse);
    }
}

impl Default for CollisionSolver {
    fn default() -> Self {
        Self::new()
    }
}