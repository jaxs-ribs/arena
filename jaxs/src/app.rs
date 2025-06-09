//! # JAXS Application Logic
//!
//! This module orchestrates the main simulation loop, integrating the physics engine,
//! rendering, and live shader reloading capabilities.
//!
//! The primary function, [`run`], is the entry point for the application's core
//! logic, called from the `main` function in `jaxs_runtime`. It is responsible for
//! driving a physics simulation and, when enabled, visualizing the simulation in
//! real-time.
//!
//! ## Features
//!
//! This crate can be compiled with the `render` feature, which enables a graphical
//! front-end for the simulation. When this feature is enabled, the application
//! will open a window to display the state of the physics simulation. The `run`
//! function will also initialize a file watcher for the `shaders/` directory,
//! allowing for hot-reloading of WGSL shader files. This is particularly useful
//! for development, as it allows for immediate feedback on shader code changes.
//!
//! If the `render` feature is not enabled, the simulation runs in a headless
//! mode, without any graphical output. This is useful for running simulations
//! on servers or in environments where a GUI is not available.

use anyhow::Result;
use physics::PhysicsSim;

#[cfg(feature = "render")]
use render::Renderer;

use crate::watcher;

/// Run the main physics simulation loop.
///
/// When `enable_render` is `true` and the crate is compiled with the `render`
/// feature, a `Renderer` window displays the positions of the
/// simulated spheres. WGSL files in the `shaders/` directory are watched for
/// changes so compute pipelines can be reloaded on the fly through
/// [`crate::watcher`].
///
/// The loop steps a [`physics::PhysicsSim`] containing a single sphere and logs
/// progress every few frames.
///
/// # Errors
///
/// Returns any error produced by the physics engine, renderer or file watcher.
pub fn run(enable_render: bool) -> Result<()> {
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
    let mut renderer = if enable_render {
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