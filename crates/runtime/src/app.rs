use anyhow::Result;
use physics::PhysicsSim;

#[cfg(feature = "render")]
use render::Renderer;

use crate::watcher;

pub fn run(_enable_render: bool) -> Result<()> {
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

    #[cfg(feature = "render")]
    let mut renderer = if _enable_render {
        Some(Renderer::new()?)
    } else {
        None
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

    #[allow(unused_mut)]
    let mut should_continue = true;
    for i in 0..num_steps {
        if let Err(e) = sim.run(dt, 1) {
            tracing::error!("Error during simulation step {}: {:?}", i, e);
            break;
        }
        #[cfg(feature = "render")]
        if let Some(r) = renderer.as_mut() {
            r.update_spheres(&sim.spheres);
            should_continue = r.render()?;
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

    #[cfg(feature = "render")]
    if let Some(mut renderer) = renderer {
        // Create a new simulation loop that just renders the final state.
        while should_continue {
            should_continue = renderer.render()?;
            // The renderer does not have its own physics loop, so we manually update positions.
            // For this example, we'll just keep rendering the final state.
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    }

    Ok(())
}
