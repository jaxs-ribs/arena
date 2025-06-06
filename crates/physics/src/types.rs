#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: u32,
}

impl Vec3 {
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _pad: 0 }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f32,
    pub _pad: [u32; 3],
}

// This struct will be passed to the shader as uniform.
// WGSL `params: vec4<f32>` where xyz: gravity, w: dt.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhysParams {
    pub gravity: Vec3,
    pub dt: f32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Joint {
    pub body_a: u32,
    pub body_b: u32,
    pub rest_length: f32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointParams {
    pub compliance: f32,
    pub _pad: [f32; 3],
}

// Structure to return from sim.run() to satisfy the test
pub struct SphereState {
    pub pos: Vec3,
    // pub vel: Vec3, // If needed later
} 