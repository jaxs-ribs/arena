//! Simple `wgpu` renderer used for visualizing physics simulations.
//!
//! The rendering code is deliberately minimal and currently only supports a few
//! debug primitives. It is not meant to be a full-featured engine but rather a
//! convenient way to inspect the simulation state while developing algorithms.

mod renderer;
mod sdf_renderer;

pub use renderer::Renderer;
pub use sdf_renderer::SdfRenderer;
