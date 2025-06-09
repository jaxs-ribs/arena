#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]
//! Entry point for the runtime binary.
//!
//! This crate ties together the compute, physics and optional rendering
//! layers. A lightweight file watcher allows WGSL shaders to be hot reloaded
//! during development. Run with `--draw` to open a window visualising the
//! simulation.

mod app;
mod watcher;

use anyhow::Result;

/// Parses command line arguments and launches [`app::run`].
///
/// Returns any error encountered during simulation setup or execution.
fn main() -> Result<()> {
    let enable_render = std::env::args().any(|a| a == "--draw");
    app::run(enable_render)
}
