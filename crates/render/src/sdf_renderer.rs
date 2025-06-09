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
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowBuilder};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_proj_inv: [[f32; 4]; 4],
    eye: [f32; 4],
}

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SphereGpu {
    pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoxGpu {
    pos: [f32; 3],
    _pad1: f32,
    half_extents: [f32; 3],
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CylinderGpu {
    pos: [f32; 3],
    radius: f32,
    height: f32,
    _pad0: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneGpu {
    normal: [f32; 3],
    d: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneCounts {
    spheres: u32,
    boxes: u32,
    cylinders: u32,
    planes: u32,
}

pub struct SdfRenderer {
    event_loop: EventLoop<()>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
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
    last_cursor: Option<(f64, f64)>,
}

impl SdfRenderer {
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new().context("create event loop")?;
        let window = WindowBuilder::new()
            .with_title("Arena SDF Renderer")
            .build(&event_loop)
            .context("failed to create window")?;

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
            eye: Vec3::new(0.0, 5.0, 15.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
            yaw: 0.0,
            pitch: 0.0,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: spheres_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: boxes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cylinders_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: planes_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sdf.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sdf pipeline"),
            layout: Some(&pipeline_layout),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let quad: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad"),
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            event_loop,
            surface,
            device,
            queue,
            config,
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
            last_cursor: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_scene(
        &mut self,
        spheres: &[Sphere],
        boxes: &[BoxBody],
        cylinders: &[Cylinder],
        planes: &[Plane],
    ) {
        let sphere_data: Vec<SphereGpu> = spheres.iter().map(|s| SphereGpu { pos: [s.pos.x, s.pos.y, s.pos.z], _pad: 0.0 }).collect();
        let box_data: Vec<BoxGpu> = boxes
            .iter()
            .map(|b| BoxGpu {
                pos: [b.pos.x, b.pos.y, b.pos.z],
                _pad1: 0.0,
                half_extents: [b.half_extents.x, b.half_extents.y, b.half_extents.z],
                _pad2: 0.0,
            })
            .collect();
        let cyl_data: Vec<CylinderGpu> = cylinders
            .iter()
            .map(|c| CylinderGpu {
                pos: [c.pos.x, c.pos.y, c.pos.z],
                radius: c.radius,
                height: c.height,
                _pad0: [0.0; 3],
            })
            .collect();
        let plane_data: Vec<PlaneGpu> = planes
            .iter()
            .map(|p| PlaneGpu { normal: [p.normal.x, p.normal.y, p.normal.z], d: p.d })
            .collect();

        let counts = SceneCounts {
            spheres: sphere_data.len() as u32,
            boxes: box_data.len() as u32,
            cylinders: cyl_data.len() as u32,
            planes: plane_data.len() as u32,
        };
        self.queue.write_buffer(&self.counts_buffer, 0, bytemuck::bytes_of(&counts));

        if !sphere_data.is_empty() {
            let bytes = bytemuck::cast_slice(&sphere_data);
            if self.spheres_buffer.size() < bytes.len() as u64 {
                self.spheres_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("spheres"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.spheres_buffer, 0, bytes);
            }
        }
        if !box_data.is_empty() {
            let bytes = bytemuck::cast_slice(&box_data);
            if self.boxes_buffer.size() < bytes.len() as u64 {
                self.boxes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("boxes"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.boxes_buffer, 0, bytes);
            }
        }
        if !cyl_data.is_empty() {
            let bytes = bytemuck::cast_slice(&cyl_data);
            if self.cylinders_buffer.size() < bytes.len() as u64 {
                self.cylinders_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cylinders"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.cylinders_buffer, 0, bytes);
            }
        }
        if !plane_data.is_empty() {
            let bytes = bytemuck::cast_slice(&plane_data);
            if self.planes_buffer.size() < bytes.len() as u64 {
                self.planes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("planes"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.planes_buffer, 0, bytes);
            }
        }
    }

    fn update_camera(&mut self, dt: f32) {
        let speed = 5.0 * dt;
        let mut forward = Vec3::new(self.camera.yaw.cos(), 0.0, self.camera.yaw.sin());
        let right = Vec3::new(-forward.z, 0.0, forward.x);
        if self.pressed_keys.contains(&KeyCode::KeyW) {
            self.camera.eye += forward * speed;
            self.camera.target += forward * speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            self.camera.eye -= forward * speed;
            self.camera.target -= forward * speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            self.camera.eye -= right * speed;
            self.camera.target -= right * speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            self.camera.eye += right * speed;
            self.camera.target += right * speed;
        }
        forward.y = 0.0;
    }

    pub fn render(&mut self) -> Result<bool> {
        let mut exit = false;
        let start = std::time::Instant::now();
        let status = self.event_loop.pump_events(Some(Duration::ZERO), |event, elwt| {
            match &event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        exit = true;
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match event.state {
                                ElementState::Pressed => { self.pressed_keys.insert(code); }
                                ElementState::Released => { self.pressed_keys.remove(&code); }
                            }
                        }
                    }
                    WindowEvent::Resized(size) => {
                        self.config.width = size.width;
                        self.config.height = size.height;
                        self.surface.configure(&self.device, &self.config);
                        self.camera.aspect = size.width as f32 / size.height as f32;
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if *button == MouseButton::Left {
                            if *state == ElementState::Released { self.last_cursor = None; }
                        }
                    }
                    _ => {}
                },
                Event::DeviceEvent { event: DeviceEvent::MouseMotion{ delta }, .. } => {
                    if self.last_cursor.is_some() {
                        self.camera.yaw -= delta.0 as f32 * 0.002;
                        self.camera.pitch = (self.camera.pitch - delta.1 as f32 * 0.002).clamp(-1.54, 1.54);
                        let dir = Vec3::new(
                            self.camera.yaw.cos() * self.camera.pitch.cos(),
                            self.camera.pitch.sin(),
                            self.camera.yaw.sin() * self.camera.pitch.cos(),
                        );
                        self.camera.target = self.camera.eye + dir;
                    }
                }
                Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                    if self.last_cursor.is_some() {
                        self.last_cursor = Some((position.x, position.y));
                    }
                }
                Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                    if *button == MouseButton::Left {
                        if *state == ElementState::Pressed {
                            self.last_cursor = Some((0.0,0.0));
                        } else {
                            self.last_cursor = None;
                        }
                    }
                }
                _ => {}
            }
        });

        if matches!(status, PumpStatus::Exit(_)) || exit {
            return Ok(false);
        }

        let dt = start.elapsed().as_secs_f32();
        self.update_camera(dt);

        let vp = self.camera.build_view_projection_matrix();
        self.camera_uniform.view_proj = vp.to_cols_array_2d();
        self.camera_uniform.view_proj_inv = vp.inverse().to_cols_array_2d();
        self.camera_uniform.eye = [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 0.0];
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&self.camera_uniform));

        let output = self.surface.get_current_texture().context("failed to acquire surface texture")?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(true)
    }
}
