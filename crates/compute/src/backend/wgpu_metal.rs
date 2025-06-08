#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{BufferView, ComputeBackend, ComputeError, Kernel};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Arc, Mutex};
use std::{fs, path::PathBuf};

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

    fn shader_path(&self, kernel: &Kernel) -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest.join("../../shaders");
        let file = match kernel {
            Kernel::IntegrateBodies => "integrate_euler.wgsl",
            Kernel::Add | Kernel::Mul | Kernel::Where => "elementwise.wgsl",
            _ => "noop.wgsl",
        };
        shaders_dir.join(file)
    }

    fn pipeline_for(&self, kernel: &Kernel) -> Result<Arc<wgpu::ComputePipeline>, ComputeError> {
        let disc = std::mem::discriminant(kernel);
        if let Some(p) = self.pipelines.lock().unwrap().get(&disc) {
            return Ok(Arc::clone(p));
        }

        let shader_src = fs::read_to_string(self.shader_path(kernel))
            .map_err(|_| ComputeError::BackendUnavailable)?;
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu_metal_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu_metal_pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });
        let arc = Arc::new(pipeline);
        self.pipelines.lock().unwrap().insert(disc, arc.clone());
        Ok(arc)
    }

    fn output_index(&self, kernel: &Kernel) -> Option<usize> {
        use Kernel::*;
        Some(match kernel {
            Add | Sub | Mul | Div | Min | Max | Clamp | Where => 2,
            Neg | Exp | Log | Sqrt | Rsqrt | Tanh | Relu | Sigmoid => 1,
            ReduceSum | ReduceMean | ReduceMax => 1,
            SegmentedReduceSum | ScatterAdd => 2,
            Gather | MatMul => 2,
            IntegrateBodies | SolveContactsPBD => 0,
            DetectContactsSDF => 2,
            ExpandInstances => 1,
            RngNormal => 0,
            SolveJointsPBD => 0,
        })
    }
}

impl ComputeBackend for WgpuMetal {
    fn dispatch(
        &self,
        shader_kernel: &Kernel,
        binds: &[BufferView],
        workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        let pipeline = self.pipeline_for(shader_kernel)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wgpu_metal_encoder") });

        let mut gpu_buffers = Vec::new();
        let mut readbacks: Vec<(wgpu::Buffer, usize, u64)> = Vec::new();
        let mut entries = Vec::new();

        for (i, view) in binds.iter().enumerate() {
            let size = view.data.len() as u64;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            if size > 0 {
                staging.slice(..).get_mapped_range_mut().copy_from_slice(&view.data);
            }
            staging.unmap();
            encoder.copy_buffer_to_buffer(&staging, 0, &buffer, 0, size);

            if Some(i) == self.output_index(shader_kernel) {
                let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                readbacks.push((readback, gpu_buffers.len(), size));
            }

            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });

            gpu_buffers.push(buffer);
        }

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute") });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        for (readback, buf_index, size) in &readbacks {
            let src = &gpu_buffers[*buf_index];
            encoder.copy_buffer_to_buffer(src, 0, readback, 0, *size);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let mut outputs = Vec::new();
        for (readback, _buf_index, size) in readbacks {
            let slice = readback.slice(..);
            pollster::block_on(slice.map_async(wgpu::MapMode::Read)).map_err(|_| ComputeError::BackendUnavailable)?;
            self.device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range().to_vec();
            readback.unmap();
            outputs.push(data[..size as usize].to_vec());
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferView, Kernel}; // Use crate:: to access items from compute crate root

    #[test]
    fn wgpu_metal_backend_add_pipeline() {
        match WgpuMetal::try_new() {
            Ok(backend) => {
                let a = vec![1.0f32, 2.0, 3.0, 4.0];
                let b = vec![5.0f32, 6.0, 7.0, 8.0];
                let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

                let a_buf = BufferView::new(bytemuck::cast_slice(&a).to_vec().into(), vec![a.len()], 4);
                let b_buf = BufferView::new(bytemuck::cast_slice(&b).to_vec().into(), vec![b.len()], 4);
                let out_placeholder: Arc<[u8]> = vec![0u8; expected.len() * 4].into();
                let out_buf = BufferView::new(out_placeholder, vec![expected.len()], 4);
                let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
                let cfg_buf = BufferView::new(cfg, vec![1], 4);

                let result = backend
                    .dispatch(&Kernel::Add, &[a_buf, b_buf, out_buf, cfg_buf], [1, 1, 1])
                    .expect("Dispatch failed");
                assert_eq!(result.len(), 1);
                let values: &[f32] = bytemuck::cast_slice(&result[0]);
                assert_eq!(values, expected.as_slice());
            }
            Err(e) => panic!("WgpuMetal::try_new() failed on macOS: {e:?}")
        }
    }
}
