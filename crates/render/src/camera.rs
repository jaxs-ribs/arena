//! Camera and controller for first-person navigation
//!
//! This module provides a simple FPS-style camera with keyboard and mouse controls
//! for navigating the 3D scene.

use glam::{Mat4, Vec3};
use winit::event::{DeviceEvent, ElementState};
use winit::keyboard::KeyCode;

/// Simple first person camera used by the renderer
pub struct Camera {
    /// Camera position
    pub eye: Vec3,
    /// Up vector
    pub up: Vec3,
    /// Render target aspect ratio
    pub aspect: f32,
    /// Field of view in radians
    pub fovy: f32,
    /// Near clipping plane distance
    pub znear: f32,
    /// Far clipping plane distance
    pub zfar: f32,
    /// Horizontal rotation of the camera
    pub yaw: f32,
    /// Vertical rotation of the camera
    pub pitch: f32,
    /// Camera movement speed
    pub speed: f32,
    /// Mouse sensitivity for look
    pub sensitivity: f32,
}

impl Camera {
    /// Create a new camera with default settings
    pub fn new(width: u32, height: u32) -> Self {
        // Position camera to view the CartPole scene
        // CartPoles are at y=0, spread from x=-2 to x=2, z=-1 to z=1
        let eye = Vec3::new(5.0, 3.0, 8.0);     // Offset to see the scene at an angle
        let target = Vec3::new(0.0, 1.0, 0.0);  // Look at the middle of the CartPoles
        let forward = (target - eye).normalize();
        let yaw = forward.x.atan2(forward.z);
        let pitch = forward.y.asin();

        Self {
            eye,
            up: Vec3::Y,
            aspect: width as f32 / height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
            yaw,
            pitch,
            speed: 10.0,
            sensitivity: 1.0,
        }
    }

    /// Update aspect ratio when window is resized
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    /// Get the camera's forward direction vector
    pub fn forward(&self) -> Vec3 {
        let yaw_quat = glam::Quat::from_axis_angle(Vec3::Y, self.yaw);
        let pitch_quat = glam::Quat::from_axis_angle(Vec3::X, self.pitch);
        let orientation = yaw_quat * pitch_quat;
        orientation * Vec3::NEG_Z
    }

    /// Get the camera's right direction vector
    pub fn right(&self) -> Vec3 {
        let yaw_quat = glam::Quat::from_axis_angle(Vec3::Y, self.yaw);
        let pitch_quat = glam::Quat::from_axis_angle(Vec3::X, self.pitch);
        let orientation = yaw_quat * pitch_quat;
        orientation * Vec3::X
    }

    /// Computes a view projection matrix from the camera parameters
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let yaw_quat = glam::Quat::from_axis_angle(Vec3::Y, self.yaw);
        let pitch_quat = glam::Quat::from_axis_angle(Vec3::X, self.pitch);
        let orientation = yaw_quat * pitch_quat;
        let forward = orientation * Vec3::NEG_Z;

        let target = self.eye + forward;

        let view = Mat4::look_at_rh(self.eye, target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

/// First person camera controller for handling input
pub struct CameraController {
    /// Movement speed multiplier
    speed: f32,
    /// Mouse look sensitivity
    sensitivity: f32,
    /// Movement state flags
    pub is_forward_pressed: bool,
    pub is_backward_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
    pub is_up_pressed: bool,
    pub is_down_pressed: bool,
}

impl CameraController {
    /// Create a new camera controller
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    /// Process keyboard events to update movement state
    pub fn process_keyboard(&mut self, keycode: KeyCode, state: ElementState) {
        let is_pressed = state == ElementState::Pressed;
        match keycode {
            KeyCode::KeyW => self.is_forward_pressed = is_pressed,
            KeyCode::KeyA => self.is_left_pressed = is_pressed,
            KeyCode::KeyS => self.is_backward_pressed = is_pressed,
            KeyCode::KeyD => self.is_right_pressed = is_pressed,
            KeyCode::Space => self.is_up_pressed = is_pressed,
            KeyCode::ShiftLeft => self.is_down_pressed = is_pressed,
            _ => (),
        }
    }

    /// Process mouse motion events to update camera look direction
    pub fn process_mouse(&self, camera: &mut Camera, delta_x: f64, delta_y: f64) {
        camera.yaw -= delta_x as f32 * self.sensitivity * 0.001;
        camera.pitch -= delta_y as f32 * self.sensitivity * 0.001;
        // Clamp pitch to prevent camera flipping
        camera.pitch = camera.pitch.clamp(-1.5, 1.5);
    }

    /// Update camera position based on current input state
    pub fn update_camera(&self, camera: &mut Camera, dt: f32) {
        let forward_dir = camera.forward();
        let right_dir = camera.right();

        let mut velocity = Vec3::ZERO;
        if self.is_forward_pressed {
            velocity += forward_dir;
        }
        if self.is_backward_pressed {
            velocity -= forward_dir;
        }
        if self.is_right_pressed {
            velocity += right_dir;
        }
        if self.is_left_pressed {
            velocity -= right_dir;
        }

        // Normalize to prevent faster diagonal movement
        if velocity.length_squared() > 0.0 {
            camera.eye += velocity.normalize() * self.speed * dt;
        }

        // Vertical movement (global axis)
        if self.is_up_pressed {
            camera.eye.y += self.speed * dt;
        }
        if self.is_down_pressed {
            camera.eye.y -= self.speed * dt;
        }
    }
}