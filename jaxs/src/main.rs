//! # JAXS Runtime
//!
//! Entry point for the JAXS runtime binary.
//!
//! This executable ties together the compute, physics, and optional rendering
//! layers. A lightweight file watcher allows WGSL shaders to be hot-reloaded
//! during development. Run with `--draw` to open a window visualizing the
//! simulation.

mod app;
mod watcher;

use anyhow::Result;

/// Entry point for the JAXS physics simulation runtime
/// 
/// Determines execution mode based on compile-time features:
/// - With `render` feature: Launches windowed simulation with visualization
/// - Without `render` feature: Runs headless simulation for testing/compute
fn main() -> Result<()> {
    let execution_mode = determine_execution_mode();
    app::run(execution_mode.should_render())
}

/// Execution mode configuration for the simulation
enum ExecutionMode {
    /// Run with graphical rendering window
    Windowed,
    /// Run without any visual output
    Headless,
}

impl ExecutionMode {
    /// Check if rendering should be enabled for this mode
    fn should_render(&self) -> bool {
        matches!(self, ExecutionMode::Windowed)
    }
}

/// Determine execution mode based on compile-time features
fn determine_execution_mode() -> ExecutionMode {
    if cfg!(feature = "render") {
        ExecutionMode::Windowed
    } else {
        ExecutionMode::Headless
    }
} 