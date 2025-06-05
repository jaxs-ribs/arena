use anyhow::{Context, Result};
use physics::Sphere;
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use std::time::Duration;
use winit::event_loop::{EventLoop};
use winit::window::WindowBuilder;
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};

pub struct Renderer<'w> {
    event_loop: EventLoop<()>,
    #[allow(dead_code)]
    window: winit::window::Window,
    surface: wgpu::Surface<'w>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[allow(dead_code)]
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sphere_buffer: wgpu::Buffer,
    sphere_capacity: u64,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    num_spheres: u32,
}

impl<'w> Renderer<'w> {
    pub fn new() -> Result<Renderer<'static>> {
        let event_loop = EventLoop::new().context("create event loop")?;
        let window = WindowBuilder::new()
            .with_title("Arena Renderer")
            .build(&event_loop)
            .context("failed to create window")?;

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
                label: Some("Renderer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .context("failed to request device")?;

        let size = window.inner_size();
        let formats = surface.get_capabilities(&adapter).formats;
        let format = formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
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
                }
            ],
        });

        let sphere_capacity = 0;

        Ok(Renderer {
            event_loop,
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            sphere_buffer,
            sphere_capacity,
            bind_group,
            bind_group_layout,
            num_spheres: 0,
        })
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
                    }
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

    pub fn render(&mut self) -> Result<()> {
        let status = self.event_loop.pump_events(Some(Duration::ZERO), |event, elwt| {
            if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = &event {
                elwt.exit();
            }
        });
        if matches!(status, PumpStatus::Exit(_)) {
            return Ok(());
        }

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
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..6, 0..self.num_spheres);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

