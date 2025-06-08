#![deny(clippy::all, clippy::pedantic)]

use std::sync::Arc;
use thiserror::Error;

mod cpu_backend;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

pub mod kernels;
pub mod layout;

pub use cpu_backend::CpuBackend;
#[cfg(feature = "gpu")]
pub use wgpu_backend::WgpuBackend;

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
    DetectContactsBox,
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

    pub fn name(&self) -> &'static str {
        match self {
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
        }
    }

    #[cfg(feature = "gpu")]
    pub fn to_shader_source(&self) -> &'static str {
        match self {
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
        }
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

/// The default backend for the current configuration.
///
/// This will be the `WgpuBackend` if the `gpu` feature is enabled, otherwise it will be the `CpuBackend`.
pub fn default_backend() -> Arc<dyn ComputeBackend> {
    #[cfg(feature = "gpu")]
    {
        Arc::new(WgpuBackend::new().unwrap())
    }
    #[cfg(not(feature = "gpu"))]
    {
        Arc::new(CpuBackend::new())
    }
}
