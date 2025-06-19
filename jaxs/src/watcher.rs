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
use tracing::{info, error};

// File system watching configuration
const SHADER_DIRECTORY: &str = "shaders";
const SHADER_EXTENSION: &str = "wgsl";

/// Handler for shader file modification events
/// 
/// Currently logs changes for debugging. In production, this will
/// trigger pipeline recompilation in the rendering system.
struct ShaderChangeHandler;

impl ShaderChangeHandler {
    /// Process a shader file change event
    fn handle_change(path: &Path) {
        info!(
            "Shader file modified: {:?} - Hot reload triggered (placeholder)",
            path.file_name().unwrap_or_default()
        );
        
        // TODO: Implement actual shader recompilation:
        // 1. Validate shader syntax
        // 2. Signal renderer to recompile pipeline
        // 3. Handle compilation errors gracefully
    }
    
    /// Check if a path represents a shader file we should watch
    fn is_shader_file(path: &Path) -> bool {
        path.extension()
            .map(|ext| ext == SHADER_EXTENSION)
            .unwrap_or(false)
    }
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
    info!("Initializing shader hot-reload watcher...");

    let mut watcher = create_file_watcher()?;
    
    start_watching_shaders(&mut watcher)?;
    
    info!("Shader watcher active - monitoring '{}' directory", SHADER_DIRECTORY);
    Ok(watcher)
}

/// Create the file system watcher with appropriate event handlers
fn create_file_watcher() -> Result<RecommendedWatcher> {
    notify::recommended_watcher(handle_file_event)
        .map_err(|e| anyhow::anyhow!("Failed to create file watcher: {}", e))
}

/// Configure watcher to monitor the shader directory
fn start_watching_shaders(watcher: &mut RecommendedWatcher) -> Result<()> {
    let shader_path = Path::new(SHADER_DIRECTORY);
    
    // Verify directory exists before watching
    if !shader_path.exists() {
        error!("Shader directory '{}' not found", SHADER_DIRECTORY);
        return Err(anyhow::anyhow!("Shader directory not found"));
    }
    
    watcher.watch(shader_path, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("Failed to watch shader directory: {}", e))
}

/// Handle file system events from the watcher
fn handle_file_event(result: notify::Result<Event>) {
    match result {
        Ok(event) => process_file_event(event),
        Err(e) => error!("File watcher error: {:?}", e),
    }
}

/// Process a file system event, filtering for relevant shader changes
fn process_file_event(event: Event) {
    // Only care about file modifications and creations
    if !event.kind.is_modify() && !event.kind.is_create() {
        return;
    }
    
    // Process each affected path
    for path in &event.paths {
        if ShaderChangeHandler::is_shader_file(path) {
            ShaderChangeHandler::handle_change(path);
        }
    }
} 