use glam::{Mat4, Vec3};
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

pub trait CameraController {
    fn update_camera(&mut self, camera: &mut Camera);
    fn process_events(&mut self, event: &WindowEvent) -> bool;
    fn process_device_events(&mut self, event: &DeviceEvent) -> bool;
}

pub struct DroneController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    mouse_sensitivity: f32,
    is_mouse_look_active: bool,
    yaw: f32,
    pitch: f32,
}

impl DroneController {
    pub fn new(speed: f32, mouse_sensitivity: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            mouse_sensitivity,
            is_mouse_look_active: false,
            yaw: -90.0,
            pitch: 0.0,
        }
    }
}

impl CameraController for DroneController {
    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state,
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.is_mouse_look_active = *state == ElementState::Pressed;
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    fn process_device_events(&mut self, event: &DeviceEvent) -> bool {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.is_mouse_look_active {
                let (dx, dy) = delta;
                self.yaw += *dx as f32 * self.mouse_sensitivity;
                self.pitch -= *dy as f32 * self.mouse_sensitivity;

                if self.pitch > 89.0 {
                    self.pitch = 89.0;
                }
                if self.pitch < -89.0 {
                    self.pitch = -89.0;
                }
                return true;
            }
        }
        false
    }

    fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let right = forward_norm.cross(camera.up);

        if self.is_forward_pressed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }
        if self.is_right_pressed {
            camera.eye += right * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * self.speed;
        }
        if self.is_up_pressed {
            camera.eye += camera.up * self.speed;
        }
        if self.is_down_pressed {
            camera.eye -= camera.up * self.speed;
        }

        let front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
        .normalize();
        camera.target = camera.eye + front;
    }
}

pub struct OrbitController {
    angle: f32,
    distance: f32,
}

impl OrbitController {
    pub fn new(distance: f32) -> Self {
        Self { angle: 0.0, distance }
    }
}

impl CameraController for OrbitController {
    fn update_camera(&mut self, camera: &mut Camera) {
        self.angle += 0.01;
        camera.eye.x = self.angle.cos() * self.distance;
        camera.eye.z = self.angle.sin() * self.distance;
        camera.target = Vec3::ZERO;
    }

    fn process_events(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn process_device_events(&mut self, _event: &DeviceEvent) -> bool {
        false
    }
} 