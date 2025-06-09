//! # Shader Hot-Reloading
//!
//! This module provides utilities for hot-reloading Web Shading Language (WGSL)
//! shaders at runtime. This feature is particularly useful during development,
//! as it allows for immediate feedback on shader code changes without needing
//! to restart the entire application.
//!
//! The core functionality is provided by the [`start`] function, which spawns a
//! file watcher that monitors the `shaders/` directory for any changes to
//! `.wgsl` files. When a change is detected, a callback function,
//! [`reload_shader`], is invoked.
//!
//! ## Implementation Details
//!
//! The file watcher is implemented using the [`notify`](https://crates.io/crates/notify)
//! crate, which provides a cross-platform API for file system notifications.
//! The watcher is configured to run in a separate thread, ensuring that it
//! doesn't block the main simulation loop.
//!
//! Currently, the `reload_shader` function is a placeholder and only logs the
//! detected change. In a future iteration, this function will be responsible
//! for triggering the recompilation of the `wgpu` pipeline, which will apply
//! the shader changes in real-time.

use anyhow::Result;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::path::Path;
use tracing::info;

/// Placeholder callback invoked whenever a shader file changes.
///
/// In the current implementation, this function only logs the path of the modified
/// shader file. However, it is designed to be the hook for the rendering engine
/// to re-create the `wgpu` pipeline. This will allow for modified compute and
/// render shaders to take effect without restarting the application.
///
/// # Future Work
///
/// -   **Pipeline Re-creation:** Implement the logic to signal the rendering
///     engine to rebuild the `wgpu` pipeline with the new shader code.
/// -   **Error Handling:** Add robust error handling to manage cases where the
///     new shader code fails to compile.
fn reload_shader(path: &Path) {
    info!(
        "Shader changed: {:?}. Triggering reload (placeholder).",
        path
    );
    // In Phase 1, this will re-create the wgpu pipeline.
}

/// Sets up and starts the file watcher for the `shaders/` directory.
///
/// This function initializes a [`RecommendedWatcher`] from the `notify` crate
/// and configures it to monitor the `shaders/` directory recursively. The
/// watcher is set to respond to modification and creation events for files
/// with the `.wgsl` extension.
///
/// When a relevant file event is detected, the [`reload_shader`] function is
/// called with the path of the changed file.
///
/// The caller of this function is responsible for keeping the returned
/// [`RecommendedWatcher`] instance alive. If the watcher is dropped, it will
/// stop monitoring for file changes.
///
/// # Errors
///
/// This function will return an error if the file watcher cannot be initialized
/// or if it fails to start watching the `shaders/` directory. These errors are
/// propagated from the underlying `notify` crate.
pub fn start() -> Result<RecommendedWatcher> {
    info!("Initializing shader watcher...");

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if event.kind.is_modify() || event.kind.is_create() {
                for path in &event.paths {
                    if path.extension().map_or(false, |ext| ext == "wgsl") {
                        reload_shader(path);
                    }
                }
            }
        }
        Err(e) => tracing::error!("Error watching shader files: {e:?}"),
    })?;

    watcher.watch(Path::new("shaders"), RecursiveMode::Recursive)?;
    info!("Shader watcher started for 'shaders/' directory.");
    Ok(watcher)
}
