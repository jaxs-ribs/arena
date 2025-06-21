//! GPU-compatible type definitions for rendering
//!
//! This module contains all the GPU buffer structures that are used to
//! pass data to the WGSL shaders. All types must be Pod and properly aligned.

use bytemuck::{Pod, Zeroable};
use physics::{BoxBody, Cylinder, Plane, Sphere};

/// Uniform buffer that stores camera matrices for the SDF renderer
///
/// The buffer contains both the view projection matrix and its inverse as well
/// as the current eye position which are required by the ray marching shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view projection matrix used for rendering
    pub view_proj: [[f32; 4]; 4],
    /// Inverse of view_proj used to transform rays into world space
    pub view_proj_inv: [[f32; 4]; 4],
    /// Camera position in world coordinates
    pub eye: [f32; 4],
}

/// GPU representation of a sphere primitive
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SphereGpu {
    /// Transform matrix (includes position and rotation)
    pub transform: [[f32; 4]; 4],
    /// Sphere radius
    pub radius: f32,
    /// Material friction coefficient
    pub friction: f32,
    /// Material restitution coefficient
    pub restitution: f32,
    pub _pad: f32,
}

impl From<&Sphere> for SphereGpu {
    fn from(sphere: &Sphere) -> Self {
        // Get transform matrix from physics module
        let transform = physics::transform::to_transform_matrix(sphere.pos, sphere.orientation);
        
        Self {
            transform,
            radius: sphere.radius,
            friction: sphere.material.friction,
            restitution: sphere.material.restitution,
            _pad: 0.0,
        }
    }
}

/// GPU representation of a box primitive
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BoxGpu {
    /// Transform matrix (includes position and rotation)
    pub transform: [[f32; 4]; 4],
    /// Half extents of the box
    pub half_extents: [f32; 3],
    pub _pad: f32,
}

impl From<&BoxBody> for BoxGpu {
    fn from(box_body: &BoxBody) -> Self {
        // Get transform matrix from physics module
        let transform = physics::transform::to_transform_matrix(box_body.pos, box_body.orientation);
        
        Self {
            transform,
            half_extents: box_body.half_extents.into(),
            _pad: 0.0,
        }
    }
}

/// GPU representation of a cylinder primitive
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CylinderGpu {
    /// Transform matrix (includes position and rotation)
    pub transform: [[f32; 4]; 4],
    /// Cylinder radius
    pub radius: f32,
    /// Height of the cylinder
    pub height: f32,
    pub _pad: [f32; 2],
}

impl From<&Cylinder> for CylinderGpu {
    fn from(cylinder: &Cylinder) -> Self {
        // Use mesh offset to properly position the visual representation
        let transform = physics::transform::to_transform_matrix_with_offset(
            cylinder.pos, 
            cylinder.orientation,
            cylinder.mesh_offset
        );
        
        Self {
            transform,
            radius: cylinder.radius,
            height: cylinder.half_height * 2.0,
            _pad: [0.0; 2],
        }
    }
}


/// GPU representation of a plane primitive
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlaneGpu {
    /// Plane normal
    pub normal: [f32; 3],
    /// Distance from the origin
    pub d: f32,
    /// Plane extents for finite rendering
    pub extents: [f32; 2],
    pub _pad: [f32; 2],
}

impl From<&Plane> for PlaneGpu {
    fn from(plane: &Plane) -> Self {
        Self {
            normal: plane.normal.into(),
            d: plane.d,
            extents: [plane.extents.x, plane.extents.y],
            _pad: [0.0; 2],
        }
    }
}

/// Keeps track of how many primitives are currently stored in the scene buffers
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SceneCounts {
    pub spheres: u32,
    pub boxes: u32,
    pub cylinders: u32,
    pub planes: u32,
}