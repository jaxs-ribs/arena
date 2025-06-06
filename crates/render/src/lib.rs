use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use physics::{PhysicsSim, Sphere};
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Plane {
    normal: [f32; 3],
    height: f32,
}

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
    fn build_view_projection_matrix(&self) -> Mat4 {
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
    fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            inv_view_proj: Mat4::IDENTITY,
            position: [0.0; 4],
            resolution: [0.0; 2],
            _pad: [0.0; 2],
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, width: u32, height: u32) {
        self.view_proj = camera.build_view_projection_matrix();
        self.inv_view_proj = self.view_proj.inverse();
        self.position = [camera.position.x, camera.position.y, camera.position.z, 1.0];
        self.resolution = [width as f32, height as f32];
    }
}

pub struct CameraState {
    mouse_pressed: bool,
    last_mouse_pos: winit::dpi::PhysicalPosition<f64>,
}

impl CameraState {
    fn new() -> Self {
        Self {
            mouse_pressed: false,
            last_mouse_pos: (0.0, 0.0).into(),
        }
    }
}

pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[allow(dead_code)]
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sphere_buffer: wgpu::Buffer,
    sphere_capacity: u64,
    plane_buffer: wgpu::Buffer,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_state: CameraState,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    num_spheres: u32,
    sim: PhysicsSim,
}

impl Renderer {
    pub async fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let window = Arc::new(WindowBuilder::new()
            .with_title("Arena Renderer")
            .build(event_loop)
            .context("failed to create window")?);

        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&*window)?)? };
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .context("failed to get adapter")?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Renderer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .context("failed to request device")?;

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let camera = Camera {
            position: (3.0, 3.0, 5.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vec3::NEG_Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_state = CameraState::new();

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, config.width, config.height);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let plane = Plane {
            normal: [0.0, 1.0, 0.0],
            height: 0.0,
        };

        let plane_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Buffer"),
            contents: bytemuck::cast_slice(&[plane]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
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

        let sphere_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spheres"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sphere_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: plane_buffer.as_entire_binding(),
                },
            ],
        });

        let sphere_capacity = 0;

        let sim = PhysicsSim::new_single_sphere(10.0);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            sphere_buffer,
            sphere_capacity,
            plane_buffer,
            camera,
            camera_uniform,
            camera_buffer,
            camera_state,
            bind_group,
            bind_group_layout,
            num_spheres: 0,
            sim,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyR),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.sim.reset();
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left {
                    self.camera_state.mouse_pressed = *state == winit::event::ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.camera_state.mouse_pressed {
                    let dx = position.x - self.camera_state.last_mouse_pos.x;
                    let dy = position.y - self.camera_state.last_mouse_pos.y;

                    let rotation_x = Mat4::from_rotation_y(dx as f32 * 0.005);
                    let rotation_y = Mat4::from_rotation_x(dy as f32 * 0.005);

                    self.camera.position = (rotation_x * rotation_y).transform_point3(self.camera.position);
                }
                self.camera_state.last_mouse_pos = *position;
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y * 0.1,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.02,
                };
                self.camera.position += self.camera.position.normalize() * scroll;
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self) {
        let dt = 0.01_f32;
        if let Err(e) = self.sim.run(dt, 1) {
            eprintln!("Error during simulation step: {:?}", e);
        }
        let spheres = self.sim.spheres.clone();
        self.update_spheres(&spheres);
    }

    pub fn update_spheres(&mut self, spheres: &[Sphere]) {
        let sphere_data = bytemuck::cast_slice(spheres);
        let required_bytes = sphere_data.len() as u64;

        if required_bytes > self.sphere_capacity {
            self.sphere_buffer.destroy();
            self.sphere_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spheres"),
                size: required_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.sphere_capacity = required_bytes;
            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sphere_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.plane_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        if required_bytes > 0 {
            self.queue.write_buffer(
                &self.sphere_buffer,
                0,
                sphere_data,
            );
        }
        self.num_spheres = spheres.len() as u32;
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.camera_uniform.update_view_proj(&self.camera, self.config.width, self.config.height);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let output = self.surface.get_current_texture()?;
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
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn run() -> Result<()> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut state = pollster::block_on(Renderer::new(&event_loop))?;
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size()),
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                // All other errors (Outdated, Timeout) should be resolved by the next frame
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                state.window().request_redraw();
            }
            _ => {}
        }
    })?;
    Ok(())
}

