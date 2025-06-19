//! # Physics Simulation Builder
//! 
//! This module provides builder functions for adding objects and constraints
//! to the physics simulation. It handles the creation of rigid bodies,
//! joints, and other physics entities.

use crate::types::{
    Material, Vec2, Vec3, Sphere, BoxBody, Cylinder, Plane,
    Joint, RevoluteJoint, PrismaticJoint, BallJoint, FixedJoint,
};
use crate::PhysicsSim;

/// Builder methods for adding rigid bodies to the simulation
impl PhysicsSim {
    /// Add a sphere with default material properties
    pub fn add_sphere(&mut self, pos: Vec3, vel: Vec3, radius: f32) -> usize {
        self.add_sphere_with_material(pos, vel, radius, Material::default())
    }

    /// Add a sphere with custom material properties
    pub fn add_sphere_with_material(
        &mut self, 
        pos: Vec3, 
        vel: Vec3, 
        radius: f32, 
        material: Material
    ) -> usize {
        let mass = calculate_sphere_mass(radius, material.density);
        self.add_sphere_with_mass_and_material(pos, vel, radius, mass, material)
    }

    /// Add a sphere with explicit mass and material
    pub fn add_sphere_with_mass_and_material(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        radius: f32,
        mass: f32,
        material: Material,
    ) -> usize {
        let sphere = Sphere {
            pos,
            prev_pos: pos,
            vel,
            radius,
            mass,
            material,
        };
        self.spheres.push(sphere);
        self.params.forces.push([0.0, 0.0]);
        self.spheres.len() - 1
    }

    /// Add a box-shaped rigid body
    pub fn add_box(&mut self, pos: Vec3, half_extents: Vec3, vel: Vec3) -> usize {
        let mass = calculate_box_mass(half_extents, 1.0); // Default density
        let box_body = BoxBody {
            pos,
            vel,
            half_extents,
            mass,
            material: Material::default(),
        };
        self.boxes.push(box_body);
        self.boxes.len() - 1
    }

    /// Add a cylindrical rigid body
    pub fn add_cylinder(
        &mut self,
        pos: Vec3,
        radius: f32,
        half_height: f32,
        vel: Vec3,
    ) -> usize {
        let mass = calculate_cylinder_mass(radius, half_height * 2.0, 1.0); // Default density
        let cylinder = Cylinder {
            pos,
            vel,
            radius,
            half_height,
            mass,
            material: Material::default(),
        };
        self.cylinders.push(cylinder);
        self.cylinders.len() - 1
    }

    /// Add a static plane for collision
    pub fn add_plane(&mut self, normal: Vec3, d: f32, extents: Vec2) -> usize {
        let plane = Plane {
            normal,
            d,
            extents,
            material: Material::default(),
        };
        self.planes.push(plane);
        self.planes.len() - 1
    }
}

/// Builder methods for adding constraints/joints
impl PhysicsSim {
    /// Add a simple distance constraint between two spheres
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        self.joints.push(Joint {
            body_a,
            body_b,
            rest_length,
            _pad: 0.0,
        });
    }

    /// Add a revolute (hinge) joint between two bodies
    pub fn add_revolute_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        let joint = RevoluteJoint {
            body_a_type,
            body_a_index,
            body_b_type,
            body_b_index,
            anchor,
            axis,
            angle: 0.0,
            angular_velocity: 0.0,
        };
        self.revolute_joints.push(joint);
        self.revolute_joints.len() - 1
    }

    /// Add a prismatic (sliding) joint between two bodies
    pub fn add_prismatic_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> usize {
        let joint = PrismaticJoint {
            body_a_type,
            body_a_index,
            body_b_type,
            body_b_index,
            anchor,
            axis,
            position: 0.0,
            linear_velocity: 0.0,
        };
        self.prismatic_joints.push(joint);
        self.prismatic_joints.len() - 1
    }

    /// Add a ball joint (3DOF rotation) between two bodies
    pub fn add_ball_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        anchor: Vec3,
    ) -> usize {
        let joint = BallJoint {
            body_a_type,
            body_a_index,
            body_b_type,
            body_b_index,
            anchor,
            _pad: [0.0; 3],
        };
        self.ball_joints.push(joint);
        self.ball_joints.len() - 1
    }

    /// Add a fixed joint (no relative motion) between two bodies
    pub fn add_fixed_joint(
        &mut self,
        body_a_type: u32,
        body_a_index: u32,
        body_b_type: u32,
        body_b_index: u32,
        relative_position: Vec3,
        relative_orientation: [f32; 4], // Quaternion
    ) -> usize {
        let joint = FixedJoint {
            body_a_type,
            body_a_index,
            body_b_type,
            body_b_index,
            relative_position,
            relative_orientation,
        };
        self.fixed_joints.push(joint);
        self.fixed_joints.len() - 1
    }
}

// Helper functions for mass calculations
fn calculate_sphere_mass(radius: f32, density: f32) -> f32 {
    let volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    volume * density
}

fn calculate_box_mass(half_extents: Vec3, density: f32) -> f32 {
    let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
    volume * density
}

fn calculate_cylinder_mass(radius: f32, height: f32, density: f32) -> f32 {
    let volume = std::f32::consts::PI * radius.powi(2) * height;
    volume * density
}