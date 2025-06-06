use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.position, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    position: [f32; 4],
    resolution: [f32; 2],
    _pad: [f32; 2],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            inv_view_proj: Mat4::IDENTITY,
            position: [0.0; 4],
            resolution: [0.0; 2],
            _pad: [0.0; 2],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, width: u32, height: u32) {
        self.view_proj = camera.build_view_projection_matrix();
        self.inv_view_proj = self.view_proj.inverse();
        self.position = [camera.position.x, camera.position.y, camera.position.z, 1.0];
        self.resolution = [width as f32, height as f32];
    }
}

pub struct CameraState {
    pub mouse_pressed: bool,
    pub last_mouse_pos: winit::dpi::PhysicalPosition<f64>,
}

impl CameraState {
    pub fn new() -> Self {
        Self {
            mouse_pressed: false,
            last_mouse_pos: (0.0, 0.0).into(),
        }
    }
} 