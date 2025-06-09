use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use physics::{BoxBody, Cylinder, Plane, Sphere};
use std::collections::HashSet;
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowBuilder};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// Uniform buffer that stores camera matrices for the SDF renderer.
///
/// The buffer contains both the view projection matrix and its inverse as well
/// as the current eye position which are required by the ray marching shader.
struct CameraUniform {
    /// Combined view projection matrix used for rendering.
    view_proj: [[f32; 4]; 4],
    /// Inverse of [`view_proj`] used to transform rays into world space.
    view_proj_inv: [[f32; 4]; 4],
    /// Camera position in world coordinates.
    eye: [f32; 4],
}

/// Simple first person camera used by [`Renderer`].
struct Camera {
    /// Camera position.
    eye: Vec3,
    /// Look at target.
    target: Vec3,
    /// Up vector.
    up: Vec3,
    /// Render target aspect ratio.
    aspect: f32,
    /// Field of view in radians.
    fovy: f32,
    /// Near clipping plane distance.
    znear: f32,
    /// Far clipping plane distance.
    zfar: f32,
    /// Horizontal rotation of the camera.
    yaw: f32,
    /// Vertical rotation of the camera.
    pitch: f32,
}

impl Camera {
    /// Computes a view projection matrix from the camera parameters.
    fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// GPU representation of a [`Sphere`].
struct SphereGpu {
    /// Sphere position.
    pos: [f32; 3],
    /// Sphere radius.
    radius: f32,
    /// Material friction coefficient.
    friction: f32,
    /// Material restitution coefficient.
    restitution: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// GPU representation of a box.
struct BoxGpu {
    /// Box centre position.
    pos: [f32; 3],
    _pad1: f32,
    /// Half extents of the box.
    half_extents: [f32; 3],
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// GPU representation of a cylinder.
struct CylinderGpu {
    /// Cylinder centre position.
    pos: [f32; 3],
    /// Cylinder radius.
    radius: f32,
    /// Height of the cylinder.
    height: f32,
    _pad0: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// GPU representation of a plane.
struct PlaneGpu {
    /// Plane normal.
    normal: [f32; 3],
    /// Distance from the origin.
    d: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// Keeps track of how many primitives are currently stored in the scene buffers.
struct SceneCounts {
    spheres: u32,
    boxes: u32,
    cylinders: u32,
    planes: u32,
}

/// A simple ray-marched renderer used for visualising signed distance fields.
pub struct Renderer {
    event_loop: EventLoop<()>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    counts_buffer: wgpu::Buffer,
    spheres_buffer: wgpu::Buffer,
    boxes_buffer: wgpu::Buffer,
    cylinders_buffer: wgpu::Buffer,
    planes_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    window: Window,
    pressed_keys: HashSet<KeyCode>,
}

impl Renderer {
    /// Create a new renderer and associated window.
    ///
    /// This sets up all GPU resources necessary for rendering the SDF scene and
    /// returns an instance ready to be updated each frame.
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new().context("create event loop")?;
        let window = WindowBuilder::new()
            .with_title("JAXS SDF Renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .with_visible(true)
            .with_resizable(true)
            .build(&event_loop)
            .context("failed to create window")?;
        
        window.set_visible(true);
        window.request_redraw();
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        let _ = window.set_cursor_visible(false);
        tracing::info!("Window created successfully");

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(&window)?;
        // Safety: surface lives as long as window
        let surface = unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("failed to get adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SDF Renderer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .context("failed to request device")?;

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera = Camera {
            eye: Vec3::new(-5.0, 6.0, 12.0),
            target: Vec3::new(0.0, 3.0, 0.0),
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
            yaw: 0.3,
            pitch: -0.3,
        };

        let view_proj = camera.build_view_projection_matrix();
        let view_proj_inv = view_proj.inverse();
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view_proj_inv: view_proj_inv.to_cols_array_2d(),
            eye: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let counts = SceneCounts { spheres: 0, boxes: 0, cylinders: 0, planes: 0 };
        let counts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counts"),
            contents: bytemuck::bytes_of(&counts),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let spheres_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spheres"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let boxes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boxes"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cylinders_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cylinders"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let planes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planes"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: boxes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cylinders_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: planes_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("renderer.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: 4 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        Ok(Self {
            event_loop,
            surface,
            device,
            queue,
            pipeline,
            vertex_buffer,
            camera,
            camera_uniform,
            camera_buffer,
            counts_buffer,
            spheres_buffer,
            boxes_buffer,
            cylinders_buffer,
            planes_buffer,
            bind_group,
            window,
            pressed_keys: HashSet::new(),
        })
    }

    /// Updates the GPU buffers with the latest positions of all scene objects.
    pub fn update_scene(
        &mut self,
        spheres: &[Sphere],
        boxes: &[BoxBody],
        cylinders: &[Cylinder],
        planes: &[Plane],
    ) {
        let sphere_data: Vec<_> = spheres
            .iter()
            .map(|s| SphereGpu {
                pos: s.pos.into(),
                radius: s.radius,
                friction: s.material.friction,
                restitution: s.material.restitution,
                _pad: [0.0; 2],
            })
            .collect();
        self.queue.write_buffer(
            &self.spheres_buffer,
            0,
            bytemuck::cast_slice(&sphere_data),
        );

        let box_data: Vec<_> = boxes
            .iter()
            .map(|b| BoxGpu {
                pos: b.pos.into(),
                half_extents: b.half_extents.into(),
                _pad1: 0.0,
                _pad2: 0.0,
            })
            .collect();
        self.queue.write_buffer(&self.boxes_buffer, 0, bytemuck::cast_slice(&box_data));

        let cylinder_data: Vec<_> = cylinders
            .iter()
            .map(|c| CylinderGpu {
                pos: c.pos.into(),
                radius: c.radius,
                height: c.height,
                _pad0: [0.0; 3],
            })
            .collect();
        self.queue
            .write_buffer(&self.cylinders_buffer, 0, bytemuck::cast_slice(&cylinder_data));

        let plane_data: Vec<_> = planes
            .iter()
            .map(|p| PlaneGpu { normal: p.normal.into(), d: p.d })
            .collect();
        self.queue.write_buffer(&self.planes_buffer, 0, bytemuck::cast_slice(&plane_data));

        let counts = SceneCounts {
            spheres: spheres.len() as u32,
            boxes: boxes.len() as u32,
            cylinders: cylinders.len() as u32,
            planes: planes.len() as u32,
        };
        self.queue
            .write_buffer(&self.counts_buffer, 0, bytemuck::bytes_of(&counts));
    }

    /// Update the camera based on keyboard input.
    fn update_camera(&mut self, dt: f32) {
        let speed = 10.0 * dt;
        let forward = self.get_forward_dir();
        let right = forward.cross(self.camera.up).normalize();
        
        for key in &self.pressed_keys {
            match key {
                KeyCode::KeyW => self.camera.eye += forward * speed,
                KeyCode::KeyS => self.camera.eye -= forward * speed,
                KeyCode::KeyA => self.camera.eye -= right * speed,
                KeyCode::KeyD => self.camera.eye += right * speed,
                KeyCode::KeyQ => self.camera.eye.y -= speed,
                KeyCode::KeyE => self.camera.eye.y += speed,
                _ => {}
            }
        }

        self.camera.target = self.camera.eye + forward;
    }

    fn get_forward_dir(&self) -> Vec3 {
        let forward = Vec3::new(
            self.camera.yaw.cos() * self.camera.pitch.cos(),
            self.camera.pitch.sin(),
            self.camera.yaw.sin() * self.camera.pitch.cos(),
        )
        .normalize();
        forward
    }

    /// Render the scene and handle window events.
    ///
    /// The method returns `Ok(false)` if the window was closed and `Ok(true)`
    /// otherwise.
    pub fn render(&mut self) -> Result<bool> {
        self.update_camera(0.016);

        let mut exit_requested = false;
        let mut events_to_handle = Vec::new();

        self.event_loop.pump_events(Some(Duration::from_millis(1)), |event, elwt| {
            if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = &event {
                exit_requested = true;
                elwt.exit();
            }
            events_to_handle.push(event);
        });

        for event in events_to_handle {
            match event {
                Event::WindowEvent { window_id, event } if window_id == self.window.id() => {
                    self.handle_window_event(event);
                }
                Event::DeviceEvent { event, .. } => self.handle_device_event(event),
                _ => {}
            }
        }


        if exit_requested || self.pressed_keys.contains(&KeyCode::Escape) {
            return Ok(false);
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let view_proj = self.camera.build_view_projection_matrix();
        let view_proj_inv = view_proj.inverse();
        self.camera_uniform.view_proj = view_proj.to_cols_array_2d();
        self.camera_uniform.view_proj_inv = view_proj_inv.to_cols_array_2d();
        self.camera_uniform.eye = [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 0.0];
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SDF Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..4, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        
        self.window.request_redraw();

        Ok(true)
    }

    fn handle_device_event(&mut self, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            let sensitivity = 0.003;
            self.camera.yaw += (delta.0 as f32) * sensitivity;
            self.camera.pitch -= (delta.1 as f32) * sensitivity;  // Invert Y for natural mouse look
            self.camera.pitch = self.camera.pitch.clamp(-1.57, 1.57); // ±90 degrees
        }
    }

    fn handle_window_event(&mut self, event: WindowEvent) {
        if let WindowEvent::KeyboardInput { event, .. } = &event {
            if event.state == ElementState::Pressed {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        self.pressed_keys.insert(KeyCode::KeyW);
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        self.pressed_keys.insert(KeyCode::KeyS);
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.pressed_keys.insert(KeyCode::KeyA);
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        self.pressed_keys.insert(KeyCode::KeyD);
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        self.pressed_keys.insert(KeyCode::KeyQ);
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        self.pressed_keys.insert(KeyCode::KeyE);
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        self.pressed_keys.insert(KeyCode::Escape);
                    }
                    _ => {}
                }
            } else if event.state == ElementState::Released {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        self.pressed_keys.remove(&KeyCode::KeyW);
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        self.pressed_keys.remove(&KeyCode::KeyS);
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.pressed_keys.remove(&KeyCode::KeyA);
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        self.pressed_keys.remove(&KeyCode::KeyD);
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        self.pressed_keys.remove(&KeyCode::KeyQ);
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        self.pressed_keys.remove(&KeyCode::KeyE);
                    }
                    _ => {}
                }
            }
        }
        if let WindowEvent::MouseInput { state, button, .. } = &event {
            if *button == MouseButton::Left && *state == ElementState::Pressed {
                let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
                let _ = self.window.set_cursor_visible(false);
            }
        }
        if let WindowEvent::Focused(focused) = &event {
            if *focused {
                let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
                let _ = self.window.set_cursor_visible(false);
            } else {
                let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                let _ = self.window.set_cursor_visible(true);
            }
        }
    }
}
