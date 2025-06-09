//! # JAXS Rendering Engine
//!
//! A simple `wgpu`-based rendering engine for visualizing physics simulations.
//!
//! This crate provides two lightweight rendering utilities for the JAXS
//! project (Just Another Exploration Substrate): [`Renderer`] and [`SdfRenderer`]. These renderers are designed to
//! be simple and efficient, providing a convenient way to visualize the state
//! of physics simulations during development and testing.
//!
//! ## Renderers
//!
//! -   **[`Renderer`]:** A basic debug renderer that can draw simple
//!     primitives like spheres, boxes, and planes. It is optimized for speed
//!     and is ideal for visualizing the basic structure of a simulation.
//! -   **[`SdfRenderer`]:** A more advanced renderer based on Signed Distance
//!     Functions (SDFs). This renderer is capable of displaying more complex
//!     shapes and scenes, but it may be more computationally intensive than
//!     the basic `Renderer`.
//!
//! ## Usage
//!
//! Both renderers can be easily integrated into tests or small binary
//! examples. To use a renderer, you first create a new instance using
//! `Renderer::new()` or `SdfRenderer::new()`. Then, in each frame of your
//! application, you can update the renderer with the latest simulation state
//! and call the `render` method to draw the scene.
//!
//! ```rust,ignore
//! use jaxs_render::Renderer;
//!
//! let mut renderer = Renderer::new()?;
//!
//! // In your game loop:
//! renderer.update_spheres(&spheres);
//! renderer.render()?;
//! ```

mod renderer;
mod sdf_renderer;

pub use renderer::Renderer;
pub use sdf_renderer::SdfRenderer;
