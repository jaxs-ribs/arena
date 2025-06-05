use anyhow::Result;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher as NotifyWatcher};
use std::path::Path;
use tracing::info;

// Placeholder for actual reload logic
fn reload_shader(path: &Path) {
    info!(
        "Shader changed: {:?}. Triggering reload (placeholder).",
        path
    );
    // In Phase 1, this will re-create the wgpu pipeline.
}

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
