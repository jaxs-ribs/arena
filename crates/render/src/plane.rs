#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Plane {
    pub normal: [f32; 3],
    pub height: f32,
} 