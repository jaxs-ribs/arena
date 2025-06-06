#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{BufferView, ComputeBackend, ComputeError, Kernel};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Arc, Mutex};

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
        _binds: &[BufferView],
        _workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        // Placeholder for all new kernels to make it compile.
        // Specific WGPU logic for each kernel will be implemented later via TDD.
        match shader_kernel {
            Kernel::Add
            | Kernel::Sub
            | Kernel::Mul
            | Kernel::Div
            | Kernel::Neg
            | Kernel::Exp
            | Kernel::Log
            | Kernel::Sqrt
            | Kernel::Rsqrt
            | Kernel::Tanh
            | Kernel::Relu
            | Kernel::Sigmoid
            | Kernel::Min
            | Kernel::Max
            | Kernel::Clamp
            | Kernel::Where => {
                eprintln!("WgpuMetal::dispatch for element-wise op {:?} - placeholder, returning Ok(Vec::new())", shader_kernel);
                Ok(Vec::new())
            }
            Kernel::ReduceSum
            | Kernel::ReduceMean
            | Kernel::ReduceMax
            | Kernel::SegmentedReduceSum
            | Kernel::ScatterAdd => {
                eprintln!("WgpuMetal::dispatch for reduction op {:?} - placeholder, returning BackendUnavailable", shader_kernel);
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::Gather => {
                eprintln!("WgpuMetal::dispatch for Gather - placeholder, returning BackendUnavailable");
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::MatMul => {
                eprintln!("WgpuMetal::dispatch for MatMul - placeholder, returning BackendUnavailable");
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::IntegrateBodies | Kernel::DetectContactsSDF | Kernel::SolveContactsPBD => {
                eprintln!("WgpuMetal::dispatch for physics op {:?} - placeholder, returning BackendUnavailable", shader_kernel);
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::SolveJointsPBD => {
                eprintln!("WgpuMetal::dispatch for SolveJointsPBD - placeholder, returning BackendUnavailable");
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::ExpandInstances => {
                eprintln!("WgpuMetal::dispatch for ExpandInstances - placeholder, returning BackendUnavailable");
                Err(ComputeError::BackendUnavailable)
            }
            Kernel::RngNormal => {
                eprintln!("WgpuMetal::dispatch for RngNormal - placeholder, returning BackendUnavailable");
                Err(ComputeError::BackendUnavailable)
            }
        }
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
}
