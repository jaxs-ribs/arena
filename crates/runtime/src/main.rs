#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::unnecessary_wraps)]

mod watcher;

use anyhow::Result;
use physics::PhysicsSim;

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

    tracing::info!("Initializing physics simulation...");
    let mut sim = PhysicsSim::new_single_sphere(10.0);
    let dt = 0.01_f32;
    let num_steps = 200;

    tracing::info!(
        "Starting simulation loop for {} steps with dt = {}...",
        num_steps,
        dt
    );
    for i in 0..num_steps {
        if let Err(e) = sim.run(dt, 1) {
            tracing::error!("Error during simulation step {}: {:?}", i, e);
            break;
        }
        if (i + 1) % 50 == 0 {
            if !sim.spheres.is_empty() {
                tracing::info!(
                    "Simulation step {} complete. Sphere_y: {}",
                    i + 1,
                    sim.spheres[0].pos.y
                );
            } else {
                tracing::info!(
                    "Simulation step {} complete. No spheres to report position.",
                    i + 1
                );
            }
        }
    }

    tracing::info!("Simulation loop finished after {} steps.", num_steps);
    if !sim.spheres.is_empty() {
        tracing::info!("Final sphere position: {:?}", sim.spheres[0].pos);
    }

    Ok(())
}
