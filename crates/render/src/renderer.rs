//! Main renderer implementation
//!
//! This module provides the main Renderer struct that coordinates all rendering
//! operations, from window creation to frame presentation.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};

use crate::camera::{Camera, CameraController};
use crate::gpu_types::{CameraUniform, SceneCounts};
use crate::pipeline::{create_bind_group_layout, create_fullscreen_quad, create_render_pipeline, create_storage_buffer, BufferConfig};
use crate::scene::SceneManager;

/// Configuration for the renderer
pub struct RendererConfig {
    /// Window title
    pub title: String,
    /// Initial window width
    pub width: u32,
    /// Initial window height
    pub height: u32,
    /// Enable vsync
    pub vsync: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            title: "JAXS SDF Renderer".to_string(),
            width: 800,
            height: 600,
            vsync: true,
        }
    }
}

/// A simple ray-marched renderer used for visualising signed distance fields
pub struct Renderer {
    // Window and surface
    window: Window,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    // GPU resources
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    
    // Pipeline and rendering
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    
    // Camera system
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    controller: CameraController,
    
    // Scene management
    scene_manager: SceneManager,
}

impl Renderer {
    /// Create a new renderer with the given configuration
    pub fn new(config: RendererConfig) -> Result<(Self, EventLoop<()>)> {
        // Create window and event loop
        let event_loop = EventLoop::new().context("Failed to create event loop")?;
        let window = WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(config.width, config.height))
            .with_visible(true)
            .with_resizable(true)
            .build(&event_loop)
            .context("Failed to create window")?;
        
        // Setup mouse capture for FPS controls
        window.set_visible(true);
        window.request_redraw();
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        let _ = window.set_cursor_visible(false);
        
        // Initialize GPU
        let (surface, adapter, device, queue) = pollster::block_on(Self::init_gpu(&window))?;
        
        // Configure surface
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if config.vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::Immediate
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        
        // Create camera
        let camera = Camera::new(config.width, config.height);
        let controller = CameraController::new(camera.speed, camera.sensitivity);
        
        // Create camera uniform buffer
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
        
        // Create scene buffers
        let buffer_config = BufferConfig::default();
        let counts = SceneCounts { spheres: 0, boxes: 0, cylinders: 0, planes: 0 };
        let counts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Counts"),
            contents: bytemuck::bytes_of(&counts),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let scene_manager = SceneManager {
            spheres_buffer: create_storage_buffer(&device, "Spheres Buffer", &buffer_config),
            boxes_buffer: create_storage_buffer(&device, "Boxes Buffer", &buffer_config),
            cylinders_buffer: create_storage_buffer(&device, "Cylinders Buffer", &buffer_config),
            planes_buffer: create_storage_buffer(&device, "Planes Buffer", &buffer_config),
            counts_buffer,
        };
        
        // Create pipeline
        let bind_group_layout = create_bind_group_layout(&device);
        let pipeline = create_render_pipeline(&device, &bind_group_layout, surface_format)?;
        
        // Create bind group
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
                    resource: scene_manager.counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_manager.spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene_manager.boxes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene_manager.cylinders_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scene_manager.planes_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create vertex buffer for fullscreen quad
        let vertex_buffer = create_fullscreen_quad(&device);
        
        Ok((
            Self {
                window,
                surface,
                surface_config,
                adapter,
                device,
                queue,
                pipeline,
                vertex_buffer,
                bind_group,
                camera,
                camera_uniform,
                camera_buffer,
                controller,
                scene_manager,
            },
            event_loop,
        ))
    }
    
    /// Initialize GPU resources
    async fn init_gpu(window: &Window) -> Result<(
        wgpu::Surface<'static>,
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
    )> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        // SAFETY: We transmute the surface to 'static lifetime. This is safe because
        // the surface will live as long as the renderer, which owns the window.
        let surface = unsafe { 
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) 
        };
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find suitable GPU adapter")?;
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SDF Renderer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .context("Failed to create GPU device")?;
        
        Ok((surface, adapter, device, queue))
    }
    
    /// Handle window and input events
    pub fn handle_event(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => self.handle_window_event(event),
            Event::DeviceEvent { event, .. } => self.handle_device_event(event),
            _ => (),
        }
    }
    
    /// Handle window-specific events
    fn handle_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Resized(physical_size) => {
                self.resize(physical_size.width, physical_size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(keycode) = event.physical_key {
                    self.controller.process_keyboard(keycode, event.state);
                    
                    // Special key handling
                    if event.state == ElementState::Pressed {
                        match keycode {
                            KeyCode::Escape => {
                                // Release mouse capture
                                let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                                let _ = self.window.set_cursor_visible(true);
                            }
                            KeyCode::KeyF => {
                                // Toggle fullscreen
                                self.window.set_fullscreen(
                                    Some(winit::window::Fullscreen::Borderless(None))
                                );
                            }
                            _ => (),
                        }
                    }
                }
            }
            _ => (),
        }
    }
    
    /// Handle device events (mouse motion)
    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.controller.process_mouse(&mut self.camera, delta.0, delta.1);
        }
    }
    
    /// Resize the rendering surface
    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
            self.camera.resize(width, height);
        }
    }
    
    /// Update camera position and matrices
    fn update_camera(&mut self, dt: f32) {
        // Update camera position based on input
        self.controller.update_camera(&mut self.camera, dt);
        
        // Update camera uniform buffer
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
    }
    
    /// Update the scene with new physics data
    pub fn update_scene(
        &mut self,
        spheres: &[physics::Sphere],
        boxes: &[physics::BoxBody],
        cylinders: &[physics::Cylinder],
        planes: &[physics::Plane],
    ) {
        self.scene_manager.update(&self.queue, spheres, boxes, cylinders, planes);
    }
    
    /// Render a single frame
    pub fn render(&mut self, dt: f32) -> Result<()> {
        // Update camera
        self.update_camera(dt);
        
        // Get next frame
        let frame = self.surface.get_current_texture()
            .context("Failed to acquire next swap chain texture")?;
        
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SDF Render Pass"),
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
        
        // Submit commands
        self.queue.submit(Some(encoder.finish()));
        
        // Present frame
        frame.present();
        
        Ok(())
    }
}