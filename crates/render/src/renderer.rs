use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use physics::{BoxBody, Cylinder, Plane, Sphere};
use wgpu::util::DeviceExt;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
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
    /// Camera movement speed.
    speed: f32,
    /// Mouse sensitivity for look.
    sensitivity: f32,
}

impl Camera {
    /// Computes a view projection matrix from the camera parameters.
    fn build_view_projection_matrix(&self) -> Mat4 {
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

/// First person camera controller.
struct Controller {
    speed: f32,
    sensitivity: f32,
    forwards: bool,
    backwards: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl Controller {
    fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            forwards: false,
            backwards: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }

    fn process_events(&mut self, keycode: KeyCode, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match keycode {
            KeyCode::KeyW => self.forwards = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyS => self.backwards = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft => self.down = pressed,
            _ => (),
        }
    }

    fn update_camera(&self, camera: &mut Camera, dt: f32) {
        let yaw_quat = glam::Quat::from_axis_angle(Vec3::Y, camera.yaw);
        let pitch_quat = glam::Quat::from_axis_angle(Vec3::X, camera.pitch);
        let orientation = yaw_quat * pitch_quat;

        let forward_dir = orientation * Vec3::NEG_Z;
        let right_dir = orientation * Vec3::X;

        let mut velocity = Vec3::ZERO;
        if self.forwards {
            velocity += forward_dir;
        }
        if self.backwards {
            velocity -= forward_dir;
        }
        if self.right {
            velocity += right_dir;
        }
        if self.left {
            velocity -= right_dir;
        }

        // Normalize to prevent faster diagonal movement
        if velocity.length_squared() > 0.0 {
            camera.eye += velocity.normalize() * self.speed * dt;
        }

        // Vertical movement (global axis)
        if self.up {
            camera.eye.y += self.speed * dt;
        }
        if self.down {
            camera.eye.y -= self.speed * dt;
        }
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
    /// Plane extents for finite rendering.
    extents: [f32; 2],
    _pad: [f32; 2],
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
    surface: wgpu::Surface<'static>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    controller: Controller,
    counts_buffer: wgpu::Buffer,
    spheres_buffer: wgpu::Buffer,
    boxes_buffer: wgpu::Buffer,
    cylinders_buffer: wgpu::Buffer,
    planes_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    window: Window,
}

impl Renderer {
    /// Create a new renderer and associated window.
    ///
    /// This sets up all GPU resources necessary for rendering the SDF scene and
    /// returns an instance ready to be updated each frame.
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Result<(Self, EventLoop<()>)> {
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

        let eye = Vec3::new(-5.0, 6.0, 12.0);
        let target = Vec3::new(0.0, 3.0, 0.0);
        let forward = (target - eye).normalize();
        let yaw = forward.x.atan2(forward.z);
        let pitch = forward.y.asin();

        let camera = Camera {
            eye,
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
            yaw,
            pitch,
            speed: 10.0,
            sensitivity: 1.0,
        };

        let view_proj = camera.build_view_projection_matrix();
        let view_proj_inv = view_proj.inverse();
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view_proj_inv: view_proj_inv.to_cols_array_2d(),
            eye: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
        };

        let controller = Controller::new(camera.speed, camera.sensitivity);

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
            label: Some("SDF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("SDF Bind Group"),
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

        let shader_src = include_str!("renderer.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                -1.0_f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok((
            Self {
                surface,
                adapter,
                device,
                queue,
                pipeline,
                vertex_buffer,
                camera,
                camera_uniform,
                camera_buffer,
                controller,
                counts_buffer,
                spheres_buffer,
                boxes_buffer,
                cylinders_buffer,
                planes_buffer,
                bind_group,
                window,
            },
            event_loop,
        ))
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
                height: c.half_height * 2.0,
                _pad0: [0.0; 3],
            })
            .collect();
        self.queue
            .write_buffer(&self.cylinders_buffer, 0, bytemuck::cast_slice(&cylinder_data));

        let plane_data: Vec<_> = planes
            .iter()
            .map(|p| PlaneGpu {
                normal: p.normal.into(),
                d: p.d,
                extents: [p.extents.x, p.extents.y],
                _pad: [0.0; 2],
            })
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

    pub fn handle_event(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => {
                self.handle_window_event(event);
            }
            Event::DeviceEvent { event, .. } => {
                self.handle_device_event(event);
            }
            _ => (),
        }
    }

    fn update_camera(&mut self, dt: f32) {
        self.controller.update_camera(&mut self.camera, dt);
    }

    /// Update the camera and render a single frame.
    ///
    /// The simulation state is passed in `world` which the renderer uses to
    /// update its buffers and render the scene. The `dt` parameter is used to
    /// update the camera based on user input.
    ///
    /// # Errors
    ///
    /// Propogates any error from rendering the frame.
    pub fn render(&mut self) -> Result<()> {
        let dt = 0.016;

        self.update_camera(dt);

        let view_proj = self.camera.build_view_projection_matrix();
        let view_proj_inv = view_proj.inverse();
        self.camera_uniform.view_proj = view_proj.to_cols_array_2d();
        self.camera_uniform.view_proj_inv = view_proj_inv.to_cols_array_2d();
        self.camera_uniform.eye = [
            self.camera.eye.x,
            self.camera.eye.y,
            self.camera.eye.z,
            0.0,
        ];
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.surface.get_capabilities(&self.adapter).formats[0]),
            ..Default::default()
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));

        frame.present();

        Ok(())
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera.yaw -= delta.0 as f32 * self.controller.sensitivity * 0.001;
            self.camera.pitch -= delta.1 as f32 * self.controller.sensitivity * 0.001;
            self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
        }
    }

    fn handle_window_event(&mut self, event: &WindowEvent) {
        if let WindowEvent::Resized(physical_size) = event {
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface.get_capabilities(&self.adapter).formats[0],
                width: physical_size.width,
                height: physical_size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            self.surface.configure(&self.device, &config);
            self.camera.aspect = physical_size.width as f32 / physical_size.height as f32;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let PhysicalKey::Code(keycode) = event.physical_key {
                self.controller.process_events(keycode, event.state);
                match keycode {
                    KeyCode::Escape => {
                        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                        let _ = self.window.set_cursor_visible(true);
                    }
                    KeyCode::KeyF => {
                        if event.state == ElementState::Pressed {
                            self.window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}