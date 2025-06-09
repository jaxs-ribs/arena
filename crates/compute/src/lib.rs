#![deny(clippy::all, clippy::pedantic)]
#![cfg_attr(
    feature = "gpu",
    doc = "Unified CPU and GPU compute abstraction.

This crate defines the [`ComputeBackend`] trait along with helper types
and kernel enumerations used throughout the rest of the workspace. The
[`CpuBackend`] provides a reference implementation and tests while the
optional [`wgpu_backend::WgpuBackend`] enables GPU acceleration when the `gpu` feature
is enabled.

Most consumers should acquire a backend via [`default_backend`] and then
call [`ComputeBackend::dispatch`] with the desired [`Kernel`] and
[`BufferView`] bindings."
)]

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
    /// Indicates that the shape of a buffer does not match the requirements
    /// of the kernel.
    #[error("buffer shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    /// Indicates that the requested compute backend is not available. For
    /// example, if the `gpu` feature is not enabled and a GPU backend is
    /// requested.
    #[error("backend not available")]
    BackendUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// An enumeration of all available compute kernels.
///
/// Each variant corresponds to a specific computation that can be performed on
/// the GPU or CPU. The kernels are organized into categories based on their
/// functionality.
pub enum Kernel {
    // ## Element-wise Operations
    // These kernels perform element-wise operations on one or more input buffers.
    /// Element-wise addition of two buffers.
    Add,
    /// Element-wise subtraction of two buffers.
    Sub,
    /// Element-wise multiplication of two buffers.
    Mul,
    /// Element-wise division of two buffers.
    Div,
    /// Negates each element of a buffer.
    Neg,
    /// Calculates the exponential of each element.
    Exp,
    /// Calculates the natural logarithm of each element.
    Log,
    /// Calculates the square root of each element.
    Sqrt,
    /// Calculates the inverse square root of each element.
    Rsqrt,
    /// Calculates the hyperbolic tangent of each element.
    Tanh,
    /// Applies the Rectified Linear Unit (ReLU) activation function.
    Relu,
    /// Applies the sigmoid activation function.
    Sigmoid,
    /// Element-wise minimum of two buffers.
    Min,
    /// Element-wise maximum of two buffers.
    Max,
    /// Clamps each element of a buffer to a given range.
    Clamp,
    /// Selects elements from two buffers based on a condition buffer.
    Where,

    // ## Reductions
    // These kernels reduce a buffer to a single value.
    /// Reduces a buffer by summing all its elements.
    ReduceSum,
    /// Calculates the mean of all elements in a buffer.
    ReduceMean,
    /// Finds the maximum element in a buffer.
    ReduceMax,
    /// Performs a segmented sum reduction.
    SegmentedReduceSum,
    /// Scatters values from one buffer into another at specified indices,
    /// adding to the existing values.
    ScatterAdd,
    /// Gathers values from a buffer at specified indices.
    Gather,

    // ## Linear Algebra
    // These kernels perform linear algebra operations.
    /// Performs matrix multiplication.
    MatMul,

    // ## Physics Simulation
    // These kernels are specific to the physics simulation.
    /// Integrates the positions and velocities of rigid bodies.
    IntegrateBodies,
    /// Detects collisions between spheres.
    DetectContactsSphere,
    /// Detects collisions between boxes.
    DetectContactsBox,
    /// Detects collisions using Signed Distance Functions (SDFs).
    DetectContactsSDF,
    /// Solves contact constraints using Position-Based Dynamics (PBD).
    SolveContactsPBD,
    /// Solves joint constraints using Position-Based Dynamics (PBD).
    SolveJointsPBD,

    // ## Miscellaneous
    // These kernels perform various utility operations.
    /// Expands instances for rendering.
    ExpandInstances,
    /// Generates random numbers from a normal distribution.
    RngNormal,
    /// Adds a buffer to another with broadcasting.
    AddBroadcast,
}

impl Kernel {
    #[must_use]
    /// Returns the number of expected buffer bindings for this kernel.
    pub const fn binding_count(&self) -> u32 {
        layout::binding_count(self)
    }
}

/// A lightweight, reference-counted view over a buffer of raw bytes.
///
/// `BufferView` provides a way to pass data to and from the compute backend
/// without incurring the cost of copying large amounts of data. It consists of
/// a reference-counted slice of bytes (`data`), a `shape` that describes the
/// logical dimensions of the data, and the `element_size_in_bytes`.
///
/// The compute backend is responsible for interpreting the raw bytes according
/// to the shape and element size.
#[derive(Clone)]
pub struct BufferView {
    /// The raw byte data of the buffer.
    pub data: Arc<[u8]>,
    /// The logical dimensions of the buffer (e.g., `[height, width]` for a 2D
    /// matrix).
    pub shape: Vec<usize>,
    /// The size in bytes of a single element in the buffer.
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

/// A trait that defines the interface for a compute backend.
///
/// A compute backend is responsible for executing compute kernels on a
/// specific device, such as a CPU or GPU. The `ComputeBackend` trait provides
/// a unified interface for dispatching kernels, regardless of the underlying
/// hardware.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Dispatches a compute kernel for execution.
    ///
    /// # Arguments
    ///
    /// * `kernel`: The [`Kernel`] to be executed.
    /// * `binds`: A slice of [`BufferView`]s that represent the input and
    ///   output buffers for the kernel. The layout and order of the buffers
    ///   are specific to each kernel and are defined in the [`layout`]
    ///   module.
    /// * `workgroups`: The number of workgroups to launch for the kernel, as a
    ///   3D array `[x, y, z]`.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec` of `Vec<u8>`, where each inner `Vec` holds
    /// the byte data of an output buffer. The order of the output buffers is
    /// determined by the kernel.
    ///
    /// An error is returned if the dispatch fails, for example, due to a
    /// shape mismatch in the input buffers or an issue with the backend.
    fn dispatch(
        &self,
        shader: &Kernel,
        binds: &[BufferView],
        workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError>;
}

/// Returns the default compute backend for the current build configuration.
///
/// If the `gpu` feature is enabled, this function returns a `WgpuBackend`.
/// Otherwise, it returns a `CpuBackend`.
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
