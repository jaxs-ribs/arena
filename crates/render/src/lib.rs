//! Simple `wgpu` renderer used for visualizing physics simulations.
//!
//! This crate exposes two small rendering utilities used throughout the project:
//! [`Renderer`], which draws a few basic debug primitives such as cubes and
//! planes, and [`SdfRenderer`], an SDF based renderer capable of displaying more
//! complex shapes.  Both implementations focus on simplicity and are not meant
//! to be full engines.  They merely provide a convenient way to inspect the
//! state of simulations while algorithms are being developed.
//!
//! The modules are intentionally minimal and avoid abstraction in favour of
//! clarity.  They can be spun up quickly in tests or small binary examples with
//! `Renderer::new()` or `SdfRenderer::new()` and then updated every frame using
//! the provided APIs.

mod renderer;
mod sdf_renderer;

pub use renderer::Renderer;
pub use sdf_renderer::SdfRenderer;
