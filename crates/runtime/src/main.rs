#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]
//! Entry point for running simulations and tests.
//!
//! This binary wires together the compute, physics and optional rendering
//! crates. Pass `--draw` to enable a real time window showing the simulation.

mod app;
mod watcher;

use anyhow::Result;

fn main() -> Result<()> {
    let enable_render = std::env::args().any(|a| a == "--draw");
    app::run(enable_render)
}
