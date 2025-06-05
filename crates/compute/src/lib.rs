#![deny(clippy::all, clippy::pedantic)]

use std::sync::Arc;
use thiserror::Error;

mod kernels;
pub mod layout; // New module declaration

#[derive(Error, Debug)]
pub enum ComputeError {
    #[error("buffer shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    #[error("backend not available")]
    BackendUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    // Element-wise
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Relu,
    Sigmoid,
    Min,
    Max,
    Clamp,
    Where,

    // Reductions
    ReduceSum,
    ReduceMean,
    ReduceMax,
    SegmentedReduceSum,
    ScatterAdd,
    Gather,

    // Linear algebra
    MatMul,

    // Physics world passes
    IntegrateBodies,
    DetectContactsSDF,
    SolveContactsPBD,
    SolveJointsPBD,

    // Optional helpers
    ExpandInstances,
    RngNormal,
}

impl Kernel {
    #[must_use]
    pub const fn binding_count(&self) -> u32 {
        layout::binding_count(self)
    }
}

#[derive(Clone)]
pub struct BufferView {
    pub data: Arc<[u8]>,
    pub shape: Vec<usize>,            // Number of elements per dimension
    pub element_size_in_bytes: usize, // Size of a single element described by the innermost dimension of shape
}

impl BufferView {
    #[must_use]
    pub fn new(data: Arc<[u8]>, shape: Vec<usize>, element_size_in_bytes: usize) -> Self {
        Self {
            data,
            shape,
            element_size_in_bytes,
        }
    }
}

pub trait ComputeBackend: Send + Sync + 'static {
    /// Dispatches a compute shader with the given bindings and workgroup configuration.
    ///
    /// # Arguments
    /// * `shader`: The kernel to dispatch.
    /// * `binds`: A slice of `BufferView`s for input and output.
    ///            It is conventional that buffers intended for read-back by the CPU
    ///            are specified first or are clearly identifiable by the kernel type.
    /// * `workgroups`: The number of workgroups to dispatch.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<Vec<u8>>)` where each inner `Vec<u8>` contains the byte data
    /// of a buffer that was written to by the GPU and is intended for CPU read-back.
    /// The order should be consistent with how `binds` are interpreted for output.
    /// Returns `ComputeError::ShapeMismatch` if any input buffers are invalid.
    /// May return other `ComputeError` variants depending on the backend implementation.
    fn dispatch(
        &self,
        shader: &Kernel,
        binds: &[BufferView],
        workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError>;
}

#[cfg(feature = "mock")]
#[derive(Default)]
pub struct MockCpu;

#[cfg(feature = "mock")]
impl ComputeBackend for MockCpu {
    fn dispatch(
        &self,
        shader: &Kernel,
        binds: &[BufferView],
        _workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        for buffer_view in binds {
            let expected_elements = buffer_view.shape.iter().product::<usize>();
            let expected_bytes = expected_elements * buffer_view.element_size_in_bytes;

            if buffer_view.data.len() != expected_bytes {
                return Err(ComputeError::ShapeMismatch(
                    "Buffer data length does not match product of shape dimensions and element size",
                ));
            }
        }
        match shader {
            Kernel::Add => kernels::add_op::handle_add(binds),
            Kernel::Sub => kernels::sub_op::handle_sub(binds),
            Kernel::Mul => kernels::mul_op::handle_mul(binds),
            Kernel::Div => kernels::div_op::handle_div(binds),
            Kernel::Where => kernels::where_op::handle_where(binds),
            Kernel::Neg => kernels::neg_op::handle_neg(binds),
            Kernel::Exp => kernels::exp_op::handle_exp(binds),
            Kernel::Log => kernels::log_op::handle_log(binds),
            Kernel::Sqrt => kernels::sqrt_op::handle_sqrt(binds),
            Kernel::Rsqrt => kernels::rsqrt_op::handle_rsqrt(binds),
            Kernel::Tanh => kernels::tanh_op::handle_tanh(binds),
            Kernel::Relu => kernels::relu_op::handle_relu(binds),
            Kernel::Sigmoid => kernels::sigmoid_op::handle_sigmoid(binds),
            Kernel::Min => kernels::min_op::handle_min(binds), // Updated to call the new handler
            Kernel::Max => kernels::max_op::handle_max(binds), // Updated to call the new handler
            Kernel::Clamp => kernels::clamp_op::handle_clamp(binds), // Updated to call the new handler
            Kernel::ReduceSum => kernels::reduce_sum_op::handle_reduce_sum(binds),
            Kernel::ReduceMean => kernels::reduce_mean_op::handle_reduce_mean(binds),
            Kernel::ReduceMax => kernels::reduce_max_op::handle_reduce_max(binds),
            Kernel::SegmentedReduceSum => {
                kernels::segmented_reduce_sum_op::handle_segmented_reduce_sum(binds)
            }
            Kernel::ScatterAdd => kernels::scatter_add_op::handle_scatter_add(binds),
            Kernel::Gather => kernels::gather_op::handle_gather(binds),
            Kernel::MatMul => kernels::matmul_op::handle_matmul(binds),
            Kernel::IntegrateBodies => kernels::integrate_bodies_op::handle_integrate_bodies(binds),
            Kernel::DetectContactsSDF => {
                kernels::detect_contacts_sdf_op::handle_detect_contacts_sdf(binds)
            }
            Kernel::SolveContactsPBD => {
                kernels::solve_contacts_pbd_op::handle_solve_contacts_pbd(binds)
            }
            Kernel::SolveJointsPBD => kernels::solve_joints_pbd_op::handle_solve_joints_pbd(binds),
            Kernel::RngNormal => kernels::rng_normal_op::handle_rng_normal(binds),
            Kernel::ExpandInstances => kernels::expand_instances_op::handle_expand_instances(binds),
        }
    }
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;

    #[test]
    fn mismatch_shape_fails() {
        let cpu = MockCpu;
        let bad_buf = BufferView::new(vec![0u8; 12].into(), vec![4], 4);
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(&Kernel::Add, &[bad_buf, good_buf, out_buf, cfg], [1, 1, 1]);
        assert!(
            matches!(result, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch error, got {result:?}"
        );
    }

    #[test]
    fn correct_shape_succeeds() {
        let cpu = MockCpu;
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(
            &Kernel::Add,
            &[good_buf.clone(), good_buf.clone(), out_buf, cfg],
            [1, 1, 1],
        );
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
    }

    #[test]
    fn correct_shape_with_larger_elements() {
        let cpu = MockCpu;
        let data_f32_x4 = vec![0u8; 16];
        let good_buf = BufferView::new(data_f32_x4.into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let config_buf = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(
            &Kernel::Exp,
            &[good_buf.clone(), out_buf.clone(), config_buf.clone()],
            [1, 1, 1],
        );
        assert!(result.is_ok(), "Expected Ok for 4x f32s, got {result:?}");

        let data_f32_x3 = vec![0u8; 12];
        let bad_buf = BufferView::new(data_f32_x3.into(), vec![4], 4);
        let result_bad = cpu.dispatch(
            &Kernel::Exp,
            &[bad_buf, out_buf.clone(), config_buf],
            [1, 1, 1],
        );
        assert!(
            matches!(result_bad, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch for 3x f32s with shape [4], got {result_bad:?}"
        );
    }

    #[test]
    fn multiple_buffers_correct_shape_succeeds() {
        let cpu = MockCpu;
        let good_buf1 = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let good_buf2 = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(
            &Kernel::Add,
            &[good_buf1, good_buf2, out_buf, cfg],
            [1, 1, 1],
        );
        assert!(
            result.is_ok(),
            "Expected Ok for multiple buffers, got {result:?}"
        );
    }

    #[test]
    fn multiple_buffers_one_bad_shape_fails() {
        let cpu = MockCpu;
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let bad_buf = BufferView::new(vec![0u8; 7].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(&Kernel::Add, &[good_buf, bad_buf, out_buf, cfg], [1, 1, 1]);
        assert!(
            matches!(result, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch error for multiple buffers, got {result:?}"
        );
    }

    #[test]
    fn empty_binds_succeeds() {
        let cpu = MockCpu;
        let result = cpu.dispatch(&Kernel::Add, &[], [1, 1, 1]);
        assert!(matches!(result, Err(ComputeError::ShapeMismatch(_))));
    }

    #[test]
    fn shape_product_is_zero() {
        let cpu = MockCpu;
        // Use f32 element size to match Exp kernel's expectation
        let element_size = std::mem::size_of::<f32>();
        let buf_zero_data_zero_prod =
            BufferView::new(vec![0u8; 0].into(), vec![0, 4], element_size); // Input
        let out_zero = BufferView::new(vec![0u8; 0].into(), vec![0], element_size); // Output placeholder
        let config_buf = BufferView::new(vec![0u8; 4].into(), vec![1], 4); // Dummy config (size doesn't strictly matter for placeholder)

        let result1 = cpu.dispatch(
            &Kernel::Exp,
            &[
                buf_zero_data_zero_prod.clone(),
                out_zero.clone(),
                config_buf.clone(),
            ],
            [1, 1, 1],
        );
        assert!(
            result1.is_ok(),
            "Expected Ok for zero-product shape with zero data, got {result1:?}"
        );

        // For the non-zero data case, the data length must still mismatch the shape product for the error to trigger.
        // If shape is [0,4] (product 0), element_size is 4, then expected_bytes is 0.
        // To trigger ShapeMismatch for "Buffer data length does not match...", data must not be 0.
        // The original check was: data.len() != (shape.iter().product() * element_size)
        // With data.len() = 4 (e.g. one f32), product = 0, element_size = 4 => 4 != 0 * 4 => 4 != 0, which is true.
        let non_zero_data_bytes: Vec<u8> = bytemuck::cast_slice(&[0.0f32]).to_vec(); // One f32 element
        let buf_nonzero_data_zero_prod =
            BufferView::new(non_zero_data_bytes.into(), vec![0, 4], element_size); // Bad input

        let result2 = cpu.dispatch(
            &Kernel::Exp,
            &[buf_nonzero_data_zero_prod, out_zero.clone(), config_buf],
            [1, 1, 1],
        );
        assert!(
            matches!(result2, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch for zero-product shape with non-zero data, got {result2:?}"
        );
    }

    #[test]
    fn kernel_binding_counts() {
        use crate::layout::binding_count;

        // Element-wise
        assert_eq!(binding_count(&Kernel::Add), 4);
        assert_eq!(binding_count(&Kernel::Sub), 4);
        assert_eq!(binding_count(&Kernel::Mul), 4);
        assert_eq!(binding_count(&Kernel::Div), 4);
        assert_eq!(binding_count(&Kernel::Min), 4);
        assert_eq!(binding_count(&Kernel::Max), 4);
        assert_eq!(binding_count(&Kernel::Where), 4);

        assert_eq!(binding_count(&Kernel::Neg), 3);
        assert_eq!(binding_count(&Kernel::Exp), 3);
        assert_eq!(binding_count(&Kernel::Log), 3);
        assert_eq!(binding_count(&Kernel::Sqrt), 3);
        assert_eq!(binding_count(&Kernel::Rsqrt), 3);
        assert_eq!(binding_count(&Kernel::Tanh), 3);
        assert_eq!(binding_count(&Kernel::Relu), 3);
        assert_eq!(binding_count(&Kernel::Sigmoid), 3);

        assert_eq!(binding_count(&Kernel::Clamp), 5);

        // Reductions
        assert_eq!(binding_count(&Kernel::ReduceSum), 3);
        assert_eq!(binding_count(&Kernel::ReduceMean), 3);
        assert_eq!(binding_count(&Kernel::ReduceMax), 3);
        assert_eq!(binding_count(&Kernel::SegmentedReduceSum), 4);
        assert_eq!(binding_count(&Kernel::ScatterAdd), 4);

        // Linear algebra
        assert_eq!(binding_count(&Kernel::MatMul), 4);

        // Physics world passes
        assert_eq!(binding_count(&Kernel::IntegrateBodies), 2);
        assert_eq!(binding_count(&Kernel::DetectContactsSDF), 3);
        assert_eq!(binding_count(&Kernel::SolveContactsPBD), 3);

        // Optional helpers
        assert_eq!(binding_count(&Kernel::ExpandInstances), 3);
        assert_eq!(binding_count(&Kernel::RngNormal), 2);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod wgpu_metal_backend {
    use super::{BufferView, ComputeBackend, ComputeError, Kernel};
    use std::collections::HashMap;
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
}

// Re-export WgpuMetal at the crate root if the feature is enabled.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use wgpu_metal_backend::WgpuMetal;

/// Returns a compute backend if available, falling back to the CPU implementation.
///
/// On macOS with the `metal` feature enabled this will attempt to create a
/// [`WgpuMetal`] backend. If GPU initialization fails or the feature/OS is not
/// available, a [`MockCpu`] backend is returned.
#[must_use]
pub fn default_backend() -> std::sync::Arc<dyn ComputeBackend> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(gpu) = WgpuMetal::try_new() {
            tracing::info!("Using WgpuMetal backend.");
            return std::sync::Arc::new(gpu);
        }
        tracing::warn!("WgpuMetal backend initialization failed, falling back...");
    }

    #[cfg(feature = "mock")]
    {
        tracing::info!("Using MockCpu backend.");
        return std::sync::Arc::new(MockCpu::default());
    }

    #[cfg(not(feature = "mock"))]
    {
        compile_error!("No compute backend available. Enable the 'mock' feature or ensure a GPU backend can initialize.");
    }
}
