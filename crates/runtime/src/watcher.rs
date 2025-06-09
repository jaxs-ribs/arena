//! Utilities for hot-reloading WGSL shaders at runtime.
//!
//! The [`start`] function spawns a [`notify`] watcher that monitors the
//! `shaders/` directory. When a `.wgsl` file changes, the path is fed into the
//! [`reload_shader`] callback which will eventually rebuild the GPU pipeline.

use anyhow::Result;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::path::Path;
use tracing::info;

/// Placeholder callback invoked whenever a shader file changes.
///
/// Currently this simply logs the change. In future iterations it will
/// recreate the `wgpu` pipeline so that modified compute and render shaders
/// take effect without restarting the application.
fn reload_shader(path: &Path) {
    info!(
        "Shader changed: {:?}. Triggering reload (placeholder).",
        path
    );
    // In Phase 1, this will re-create the wgpu pipeline.
}

/// Begin watching the `shaders/` directory for WGSL file changes.
///
/// The returned [`RecommendedWatcher`] must be kept alive by the caller to
/// continue receiving events. For every modification or creation of a `.wgsl`
/// file [`reload_shader`] is invoked.
///
/// # Errors
///
/// Propagates any errors from the underlying [`notify`] crate when setting up
/// the watcher or adding the directory to be watched.
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
