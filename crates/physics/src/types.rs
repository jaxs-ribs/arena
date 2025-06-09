#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub pos: Vec3,
    _pad1: f32,
    pub vel: Vec3,
    _pad2: f32,
    pub orientation: [f32; 4],
    pub angular_vel: Vec3,
    _pad3: f32,
}

impl Sphere {
    #[must_use]
    pub const fn new(pos: Vec3, vel: Vec3) -> Self {
        Self {
            pos,
            _pad1: 0.0,
            vel,
            _pad2: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::new(0.0, 0.0, 0.0),
            _pad3: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxShape {
    pub center: Vec3,
    _pad1: f32,
    pub half_extents: Vec3,
    _pad2: f32,
}

#[derive(Clone, Debug)]
pub struct PhysParams {
    pub gravity: Vec3,
    pub dt: f32,
    /// External forces applied per sphere (x/y components). Length must match number of spheres.
    pub forces: Vec<[f32; 2]>,
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
