//! SDF (Signed Distance Field) ray-marching renderer
//!
//! This crate provides a real-time 3D renderer for visualizing physics simulations
//! using ray marching through signed distance fields. It supports rendering of
//! spheres, boxes, cylinders, and planes with a first-person camera controller.

mod camera;
mod gpu_types;
mod pipeline;
mod renderer;
mod scene;

pub use renderer::{Renderer, RendererConfig};
