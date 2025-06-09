#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]
//! # JAXS Runtime
//!
//! Entry point for the runtime binary.
//!
//! This crate ties together the compute, physics and optional rendering
//! layers. A lightweight file watcher allows WGSL shaders to be hot reloaded
//! during development. Run with `--draw` to open a window visualising the
//! simulation.

mod app;
mod watcher;

use anyhow::Result;

/// The `main` function is the entry point of the JAXS runtime. It is responsible for:
///
/// 1.  **Parsing Command-Line Arguments:** It checks for the presence of the `--draw` flag.
/// 2.  **Launching the Application:** It calls the `app::run` function, passing a boolean value
///     that indicates whether rendering should be enabled.
///
/// The `--draw` flag enables the visualization of the simulation in a window. If the flag is
/// not present, the simulation will run in headless mode.
///
/// # Returns
///
/// This function returns a `Result<()>` which will be `Ok(())` if the simulation runs and
/// exits successfully. Any errors encountered during the setup or execution of the simulation
/// will be returned as an `Err`.
fn main() -> Result<()> {
    let enable_render = std::env::args().any(|a| a == "--draw");
    app::run(enable_render)
}
