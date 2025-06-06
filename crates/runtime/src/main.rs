#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]

mod watcher;

use anyhow::Result;

#[cfg(feature = "render")]
use render;

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

    let enable_render = std::env::args().any(|a| a == "--draw");

    if enable_render {
        #[cfg(feature = "render")]
        {
            render::run()?;
        }
    } else {
        tracing::info!("Running in headless mode. No renderer will be used.");
        // Placeholder for future headless simulation logic
    }

    Ok(())
}
