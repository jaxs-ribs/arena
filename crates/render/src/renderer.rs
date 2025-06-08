use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use physics::Sphere;
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowBuilder};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

pub struct Renderer {
    event_loop: EventLoop<()>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    _config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<[f32; 3]>,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    window: Window,
}

impl Renderer {
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new().context("create event loop")?;
        let window = WindowBuilder::new()
            .with_title("Arena Renderer")
            .build(&event_loop)
            .context("failed to create window")?;

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(&window)?;
        // We need to convince Rust that the surface is allowed to live as long as the
        // renderer, so we transmute its lifetime to `'static`.
        // This is safe because we also store the window in the renderer, and ensure
        // that it is dropped after the surface.
        let surface =
            unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("failed to get adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Renderer Device"),
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
        };

        let mut camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        };
        camera_uniform.view_proj = camera.build_view_projection_matrix().to_cols_array_2d();

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
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

        let vertices: Vec<[f32; 3]> = Vec::new();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            event_loop,
            surface,
            device,
            queue,
            _config: config,
            pipeline,
            vertex_buffer,
            vertices,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            window,
        })
    }

    pub fn update_spheres(&mut self, spheres: &[Sphere]) {
        self.vertices.clear();
        let point_size = 0.5;

        // Add a plane
        self.vertices.push([-10.0, 0.0, -10.0]);
        self.vertices.push([10.0, 0.0, -10.0]);
        self.vertices.push([10.0, 0.0, 10.0]);
        self.vertices.push([-10.0, 0.0, -10.0]);
        self.vertices.push([10.0, 0.0, 10.0]);
        self.vertices.push([-10.0, 0.0, 10.0]);

        for s in spheres {
            let x = s.pos.x;
            let y = s.pos.y;
            let z = s.pos.z;

            // Simple cube for each sphere
            // Front
            self.vertices.push([x - point_size, y - point_size, z + point_size]);
            self.vertices.push([x + point_size, y - point_size, z + point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y - point_size, z + point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y + point_size, z + point_size]);
            // Back
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z - point_size]);
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z - point_size]);
            self.vertices.push([x - point_size, y + point_size, z - point_size]);
            // Top
            self.vertices.push([x - point_size, y + point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y + point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y + point_size, z + point_size]);
            // Bottom
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y - point_size, z + point_size]);
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y - point_size, z + point_size]);
            self.vertices.push([x - point_size, y - point_size, z + point_size]);
            // Right
            self.vertices.push([x + point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x + point_size, y - point_size, z - point_size]);
            self.vertices.push([x + point_size, y + point_size, z + point_size]);
            self.vertices.push([x + point_size, y - point_size, z + point_size]);
            // Left
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x - point_size, y + point_size, z - point_size]);
            self.vertices.push([x - point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y - point_size, z - point_size]);
            self.vertices.push([x - point_size, y + point_size, z + point_size]);
            self.vertices.push([x - point_size, y - point_size, z + point_size]);
        }

        if !self.vertices.is_empty() {
            let vertex_data_bytes = bytemuck::cast_slice(&self.vertices);
            if self.vertex_buffer.size() < vertex_data_bytes.len() as u64 {
                self.vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("vertices"),
                            contents: vertex_data_bytes,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        });
            } else {
                self.queue
                    .write_buffer(&self.vertex_buffer, 0, vertex_data_bytes);
            }
        }
    }

    pub fn render(&mut self) -> Result<bool> {
        let mut exit_requested = false;
        let status = self
            .event_loop
            .pump_events(Some(Duration::ZERO), |event, elwt| {
                if let Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } = &event
                {
                    exit_requested = true;
                    elwt.exit();
                }
            });

        if matches!(status, PumpStatus::Exit(_)) || exit_requested {
            return Ok(false);
        }

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );

        let output = self
            .surface
            .get_current_texture()
            .context("failed to acquire surface texture")?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass"),
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
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..self.vertices.len() as u32, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(true)
    }
} 