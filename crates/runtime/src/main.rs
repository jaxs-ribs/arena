#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]

mod app;
mod watcher;

use anyhow::Result;

fn main() -> Result<()> {
    let enable_render = std::env::args().any(|a| a == "--draw");
    app::run(enable_render)
}
