// This file will contain the WGPU backend implementation.
// It is part of the plan to create a GPU-accelerated compute backend. 

use crate::{BufferView, ComputeBackend, ComputeError, Kernel};
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(anyhow::anyhow!("Failed to find an appropriate adapter"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

fn kernel_name(kernel: &Kernel) -> &'static str {
    match kernel {
        Kernel::Add => "add",
        Kernel::Sub => "sub",
        Kernel::Mul => "mul",
        Kernel::Div => "div",
        Kernel::Neg => "neg",
        Kernel::Exp => "exp",
        Kernel::Log => "log",
        Kernel::Sqrt => "sqrt",
        Kernel::Rsqrt => "rsqrt",
        Kernel::Tanh => "tanh",
        Kernel::Relu => "relu",
        Kernel::Sigmoid => "sigmoid",
        Kernel::Min => "min",
        Kernel::Max => "max",
        Kernel::Clamp => "clamp",
        Kernel::Where => "where",
        Kernel::ReduceSum => "reduce_sum",
        Kernel::ReduceMean => "reduce_mean",
        Kernel::ReduceMax => "reduce_max",
        Kernel::SegmentedReduceSum => "segmented_reduce_sum",
        Kernel::ScatterAdd => "scatter_add",
        Kernel::Gather => "gather",
        Kernel::MatMul => "matmul",
        Kernel::IntegrateBodies => "integrate_bodies",
        Kernel::DetectContactsSphere => "detect_contacts_sphere",
        Kernel::DetectContactsBox => "detect_contacts_box",
        Kernel::DetectContactsSDF => "detect_contacts_sdf",
        Kernel::SolveContactsPBD => "solve_contacts_pbd",
        Kernel::SolveJointsPBD => "solve_joints_pbd",
        Kernel::ExpandInstances => "expand_instances",
        Kernel::RngNormal => "rng_normal",
        Kernel::AddBroadcast => "add_broadcast",
    }
}

fn is_read_only(kernel: &Kernel, binding: u32) -> bool {
    let binding_count = crate::layout::binding_count(kernel);
    match kernel {
        Kernel::RngNormal => binding != 0,
        Kernel::ReduceMean => binding == 0 || binding == 2,
        Kernel::ReduceSum => binding == 0,
        Kernel::Neg | Kernel::Relu | Kernel::ExpandInstances => binding == 0 || binding == 2,
        Kernel::MatMul => binding == 0 || binding == 1 || binding == 3,
        Kernel::IntegrateBodies => binding == 1 || binding == 2,
        Kernel::Gather => binding == 0 || binding == 1 || binding == 3,
        Kernel::ScatterAdd => binding == 0 || binding == 1 || binding == 3,
        _ => binding < binding_count - 1,
    }
}

fn is_uniform(kernel: &Kernel, binding: u32) -> bool {
    match kernel {
        Kernel::ExpandInstances => binding == 2,
        Kernel::MatMul => binding == 3,
        Kernel::IntegrateBodies => binding == 1,
        _ => false,
    }
}

fn to_shader_source(kernel: &Kernel) -> &'static str {
    match kernel {
        Kernel::Add => include_str!("../../../shaders/add.wgsl"),
        Kernel::Sub => include_str!("../../../shaders/sub.wgsl"),
        Kernel::Mul => include_str!("../../../shaders/mul.wgsl"),
        Kernel::Div => include_str!("../../../shaders/div.wgsl"),
        Kernel::Neg => include_str!("../../../shaders/neg.wgsl"),
        Kernel::Exp => include_str!("../../../shaders/exp.wgsl"),
        Kernel::Log => include_str!("../../../shaders/log.wgsl"),
        Kernel::Sqrt => include_str!("../../../shaders/sqrt.wgsl"),
        Kernel::Rsqrt => include_str!("../../../shaders/rsqrt.wgsl"),
        Kernel::Tanh => include_str!("../../../shaders/tanh.wgsl"),
        Kernel::Relu => include_str!("../../../shaders/relu.wgsl"),
        Kernel::Sigmoid => include_str!("../../../shaders/sigmoid.wgsl"),
        Kernel::Min => include_str!("../../../shaders/min.wgsl"),
        Kernel::Max => include_str!("../../../shaders/max.wgsl"),
        Kernel::Clamp => include_str!("../../../shaders/clamp.wgsl"),
        Kernel::Where => include_str!("../../../shaders/where.wgsl"),
        Kernel::ReduceSum => include_str!("../../../shaders/reduce_sum.wgsl"),
        Kernel::ReduceMean => include_str!("../../../shaders/reduce_mean.wgsl"),
        Kernel::ReduceMax => include_str!("../../../shaders/reduce_max.wgsl"),
        Kernel::SegmentedReduceSum => include_str!("../../../shaders/segmented_reduce_sum.wgsl"),
        Kernel::ScatterAdd => include_str!("../../../shaders/scatter_add.wgsl"),
        Kernel::Gather => include_str!("../../../shaders/gather.wgsl"),
        Kernel::MatMul => include_str!("../../../shaders/matmul.wgsl"),
        Kernel::IntegrateBodies => include_str!("../../../shaders/integrate_bodies.wgsl"),
        Kernel::DetectContactsSphere => include_str!("../../../shaders/detect_contacts_sphere.wgsl"),
        Kernel::DetectContactsBox => include_str!("../../../shaders/detect_contacts_box.wgsl"),
        Kernel::DetectContactsSDF => include_str!("../../../shaders/detect_contacts_sdf.wgsl"),
        Kernel::SolveContactsPBD => include_str!("../../../shaders/solve_contacts_pbd.wgsl"),
        Kernel::SolveJointsPBD => include_str!("../../../shaders/solve_joints_pbd.wgsl"),
        Kernel::ExpandInstances => include_str!("../../../shaders/expand_instances.wgsl"),
        Kernel::RngNormal => include_str!("../../../shaders/rng_normal.wgsl"),
        Kernel::AddBroadcast => include_str!("../../../shaders/add_broadcast.wgsl"),
    }
}

impl ComputeBackend for WgpuBackend {
    fn dispatch(
        &self,
        kernel: &Kernel,
        bindings: &[BufferView],
        workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        let shader_source = to_shader_source(kernel);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kernel_name(kernel)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let mut gpu_buffers = Vec::new();
        let mut bind_group_entries = Vec::new();
        for (i, buffer_view) in bindings.iter().enumerate() {
            let buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Buffer {}", i)),
                    contents: if buffer_view.data.is_empty() {
                        &[0u8]
                    } else {
                        &buffer_view.data
                    },
                    usage: if is_uniform(kernel, i as u32) {
                        wgpu::BufferUsages::UNIFORM
                            | wgpu::BufferUsages::COPY_DST
                    } else {
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC
                    },
                });
            gpu_buffers.push(buffer);
        }

        for (i, buffer) in gpu_buffers.iter().enumerate() {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let bind_group_layout = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &(0..crate::layout::binding_count(kernel))
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if is_uniform(kernel, i as u32) {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: is_read_only(kernel, i as u32),
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            },
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        let mut output_buffers = Vec::new();
        for (i, buffer_view) in bindings.iter().enumerate() {
            if !is_read_only(kernel, i as u32) {
                let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Staging Buffer {}", i)),
                    size: buffer_view.data.len() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(
                    &gpu_buffers[i],
                    0,
                    &staging_buffer,
                    0,
                    buffer_view.data.len() as u64,
                );
                output_buffers.push(staging_buffer);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        let mut results = Vec::new();
        for buffer in output_buffers.iter() {
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = buffer_slice.get_mapped_range();
            results.push(data.to_vec());
        }

        Ok(results)
    }
} 