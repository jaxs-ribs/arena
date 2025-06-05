#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]

mod watcher;

use anyhow::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let _shader_watcher = match watcher::start() {
        Ok(watcher_instance) => {
            tracing::info!("Shader watcher started successfully.");
            Some(watcher_instance)
        }
        Err(e) => {
            tracing::error!("Failed to start shader watcher: {e:?}");
            None
        }
    };

    println!("Phase 0 boots.");
    tracing::info!("Runtime main execution finished. If watcher was started, it will run until program termination.");
    // To test the watcher, you might need to add a sleep or keep the program running longer,
    // then modify a .wgsl file in the shaders/ directory.
    std::thread::sleep(std::time::Duration::from_secs(10)); // Keep running for 10s to test watcher

    Ok(())
}
