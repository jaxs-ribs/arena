#![deny(clippy::all, clippy::pedantic)]
//! Unified CPU and GPU compute abstraction.
//!
//! This crate defines the [`ComputeBackend`] trait along with helper types
//! and kernel enumerations used throughout the rest of the workspace. The
//! [`CpuBackend`] provides a reference implementation and tests while the
//! optional [`WgpuBackend`] enables GPU acceleration when the `gpu` feature
//! is enabled.
//!
//! Most consumers should acquire a backend via [`default_backend`] and then
//! call [`ComputeBackend::dispatch`] with the desired [`Kernel`] and
//! [`BufferView`] bindings.

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
/// Errors that may occur when dispatching compute kernels.
pub enum ComputeError {
    #[error("buffer shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    #[error("backend not available")]
    BackendUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Enumeration of all compute kernels available in the crate.
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
    AddBroadcast,
}

impl Kernel {
    #[must_use]
    /// Returns the number of expected buffer bindings for this kernel.
    pub const fn binding_count(&self) -> u32 {
        layout::binding_count(self)
    }
}

#[derive(Clone)]
/// Lightweight view over a typed buffer used when dispatching kernels.
///
/// The `data` field holds a reference counted slice of raw bytes while `shape`
/// describes the logical dimensions of the buffer. `element_size_in_bytes`
/// corresponds to the innermost dimension's element size. No validation is
/// performed on construction; the [`ComputeBackend`] implementation may return
/// [`ComputeError::ShapeMismatch`] if inconsistent views are passed.
pub struct BufferView {
    /// Raw buffer contents.
    pub data: Arc<[u8]>,
    /// Number of elements per dimension.
    pub shape: Vec<usize>,
    /// Size in bytes of a single element described by the innermost dimension.
    pub element_size_in_bytes: usize,
}

impl BufferView {
    #[must_use]
    /// Creates a new buffer view over raw bytes.
    ///
    /// `shape` describes the logical tensor dimensions and `element_size_in_bytes`
    /// specifies the size of each innermost element.
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
