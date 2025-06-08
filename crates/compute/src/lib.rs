#![deny(clippy::all, clippy::pedantic)]

use std::sync::Arc;
use thiserror::Error;

mod kernels;
pub mod layout;
pub mod backend;

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
    DetectContactsSphere,
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


/// Returns a compute backend if available, falling back to the CPU implementation.
///
/// On macOS with the `metal` feature enabled this will attempt to create a
/// [`WgpuMetal`] backend. If GPU initialization fails or the feature/OS is not
/// available, a [`MockCpu`] backend is returned.
#[must_use]
pub fn default_backend() -> std::sync::Arc<dyn ComputeBackend> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(gpu) = backend::wgpu_metal::WgpuMetal::try_new() {
            tracing::info!("Using WgpuMetal backend.");
            return std::sync::Arc::new(gpu);
        }
        tracing::warn!("WgpuMetal backend initialization failed, falling back...");
    }

    #[cfg(feature = "mock")]
    {
        tracing::info!("Using MockCpu backend.");
        return std::sync::Arc::new(backend::mock_cpu::MockCpu::default());
    }

    #[cfg(not(feature = "mock"))]
    {
        compile_error!("No compute backend available. Enable the 'mock' feature or ensure a GPU backend can initialize.");
    }
}
