#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{BufferView, ComputeBackend, ComputeError, Kernel};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Arc, Mutex};
#[cfg(all(target_os = "macos", feature = "metal"))]
use wgpu::util::DeviceExt;

pub struct WgpuMetal {
    #[allow(dead_code)] // instance might not be used directly after init for basic elementwise
    instance: wgpu::Instance,
    #[allow(dead_code)] // adapter might not be used directly after init for basic Noop
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    // Pipeline cache - Mutex for interior mutability with &self in dispatch
    pipelines: Mutex<HashMap<std::mem::Discriminant<Kernel>, Arc<wgpu::ComputePipeline>>>,
}

impl WgpuMetal {
    pub fn try_new() -> Result<Self, ComputeError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            // dx12_shader_compiler: wgpu::Dx12Compiler::default(), // Removed, not in 0.19
            // flags: wgpu::InstanceFlags::default(), // Removed, not in 0.19
            // gles_minor_version: wgpu::Gles3MinorVersion::default(), // Removed, not in 0.19
            ..Default::default() // Use default for other fields
        });

        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        };

        let adapter = pollster::block_on(instance.request_adapter(&adapter_options))
            .ok_or_else(|| {
                eprintln!("Failed to find a suitable Metal adapter.");
                ComputeError::BackendUnavailable
            })?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WgpuMetal Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                // memory_hints: wgpu::MemoryHints::default(), // Removed, not in 0.19
            },
            None, // Trace path
        ))
        .map_err(|err| {
            eprintln!("Failed to request device: {err:?}");
            ComputeError::BackendUnavailable
        })?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Mutex::new(HashMap::new()),
        })
    }
}

impl ComputeBackend for WgpuMetal {
    fn dispatch(
        &self,
        shader_kernel: &Kernel,
        binds: &[BufferView],
        _workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        match shader_kernel {
            Kernel::Add => self.dispatch_binary_op(include_str!("../../../shaders/add.wgsl"), binds),
            Kernel::Sub => self.dispatch_binary_op(include_str!("../../../shaders/sub.wgsl"), binds),
            Kernel::Mul => self.dispatch_binary_op(include_str!("../../../shaders/mul.wgsl"), binds),
            Kernel::Div => self.dispatch_binary_op(include_str!("../../../shaders/div.wgsl"), binds),
            Kernel::Neg => self.dispatch_unary_op(include_str!("../../../shaders/neg.wgsl"), binds),
            Kernel::Exp => self.dispatch_unary_op(include_str!("../../../shaders/exp.wgsl"), binds),
            Kernel::Log => self.dispatch_unary_op(include_str!("../../../shaders/log.wgsl"), binds),
            _ => Err(ComputeError::BackendUnavailable),
        }
    }
}

impl WgpuMetal {
    fn dispatch_binary_op(&self, source: &str, binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
        if binds.len() < 4 {
            return Err(ComputeError::ShapeMismatch("binary elementwise expects 4 buffers"));
        }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binary_op"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::STORAGE_IN,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::STORAGE_IN2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::STORAGE_OUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::UNIFORM_SC,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("binary_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("binary_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let buffer_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_a"),
            contents: &binds[0].data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_b"),
            contents: &binds[1].data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: binds[2].data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_cfg = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cfg"),
            contents: &binds[3].data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: crate::layout::STORAGE_IN, resource: buffer_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::layout::STORAGE_IN2, resource: buffer_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::layout::STORAGE_OUT, resource: buffer_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::layout::UNIFORM_SC, resource: buffer_cfg.as_entire_binding() },
            ],
            label: Some("binary_bind_group"),
        });

        let num_elems = binds[2].data.len() / binds[2].element_size_in_bytes;
        let workgroups = ((num_elems as u32 + 255) / 256, 1, 1);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("binary_encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("binary_pass") });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: binds[2].data.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, binds[2].data.len() as u64);

        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        pollster::block_on(slice.map_async(wgpu::MapMode::Read)).map_err(|_| ComputeError::BackendUnavailable)?;
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(vec![data])
    }

    fn dispatch_unary_op(&self, source: &str, binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
        if binds.len() < 3 {
            return Err(ComputeError::ShapeMismatch("unary elementwise expects 3 buffers"));
        }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unary_op"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::STORAGE_IN,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::STORAGE_OUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: crate::layout::UNIFORM_SC,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unary_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("unary_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let buffer_in = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in"),
            contents: &binds[0].data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: binds[1].data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_cfg = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cfg"),
            contents: &binds[2].data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: crate::layout::STORAGE_IN, resource: buffer_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::layout::STORAGE_OUT, resource: buffer_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: crate::layout::UNIFORM_SC, resource: buffer_cfg.as_entire_binding() },
            ],
            label: Some("unary_bind_group"),
        });

        let num_elems = binds[1].data.len() / binds[1].element_size_in_bytes;
        let workgroups = ((num_elems as u32 + 255) / 256, 1, 1);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("unary_encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("unary_pass") });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: binds[1].data.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, binds[1].data.len() as u64);

        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        pollster::block_on(slice.map_async(wgpu::MapMode::Read)).map_err(|_| ComputeError::BackendUnavailable)?;
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(vec![data])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferView, Kernel}; // Use crate:: to access items from compute crate root

    #[test]
    fn wgpu_metal_backend_try_new_and_dispatch_noop() {
        match WgpuMetal::try_new() {
            Ok(backend) => {
                let dummy_data: Arc<[u8]> = vec![0u8; 4].into();
                // BufferView::new expects data, shape, and element_size_in_bytes
                let ok_buf = BufferView::new(dummy_data, vec![4], 1);
                let result = backend.dispatch(&Kernel::Add, &[ok_buf], [1, 1, 1]);
                assert!(
                    result.is_ok(),
                    "Dispatching Add on WgpuMetal backend failed: {result:?}"
                );
                assert!(
                    result.unwrap().is_empty(),
                    "Expected no data back from WgpuMetal Add"
                );
            }
            Err(e) => {
                // This test runs only on macOS due to cfg(all(target_os = "macos", feature = "metal")).
                // If try_new fails here, it's a significant issue with the Metal setup or wgpu.
                panic!("WgpuMetal::try_new() failed on macOS: {e:?}. Ensure Metal is available and working correctly.");
            }
        }
    }

    #[test]
    fn metal_add_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, -2.0, 0.0, 3.5, -0.5];
        let b = vec![0.5f32, 2.0, -1.0, -0.5, 10.0];
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let b_buf = BufferView::new(bytemuck::cast_slice(&b).to_vec().into(), vec![b.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Add, &[a_buf, b_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_sub_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, -2.0, 0.0, 3.5];
        let b = vec![0.5f32, 2.0, -1.0, -0.5];
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let b_buf = BufferView::new(bytemuck::cast_slice(&b).to_vec().into(), vec![b.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Sub, &[a_buf, b_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_mul_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, -2.0, 0.0, 3.5];
        let b = vec![0.5f32, 2.0, -1.0, -0.5];
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let b_buf = BufferView::new(bytemuck::cast_slice(&b).to_vec().into(), vec![b.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Mul, &[a_buf, b_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_div_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, -2.0, 1.0, 8.0];
        let b = vec![0.5f32, 2.0, 1.0, -2.0];
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / y).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let b_buf = BufferView::new(bytemuck::cast_slice(&b).to_vec().into(), vec![b.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Div, &[a_buf, b_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_neg_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, -2.0, 0.5, 3.5];
        let expected: Vec<f32> = a.iter().map(|x| -*x).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Neg, &[a_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_exp_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![0.0f32, 1.0, -1.0, 2.0];
        let expected: Vec<f32> = a.iter().map(|x| x.exp()).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Exp, &[a_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn metal_log_matches_cpu() {
        let backend = WgpuMetal::try_new().expect("Metal backend");
        let a = vec![1.0f32, std::f32::consts::E, 0.5, 2.0];
        let expected: Vec<f32> = a.iter().map(|x| x.ln()).collect();

        let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], std::mem::size_of::<f32>());
        let out_buf = BufferView::new(vec![0u8; expected.len() * std::mem::size_of::<f32>()].into(), vec![expected.len()], std::mem::size_of::<f32>());
        let cfg_buf = BufferView::new(bytemuck::cast_slice(&[0u32]).to_vec().into(), vec![1], std::mem::size_of::<u32>());

        let result = backend
            .dispatch(&Kernel::Log, &[a_buf, out_buf, cfg_buf], [1, 1, 1])
            .expect("dispatch");

        let values: &[f32] = bytemuck::cast_slice(&result[0]);
        assert_eq!(values, expected.as_slice());
    }
}
