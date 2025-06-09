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
use physics::{PhysicsSim, Vec3};
use std::time::{Duration, Instant};

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
    let mut sim = PhysicsSim::new();
    
    // Enable gravity for dynamic motion
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Ground plane at y=0
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0);
    
    // Tilted ramp plane  
    let ramp_normal = Vec3::new(0.3, 1.0, 0.0).normalize();
    sim.add_plane(ramp_normal, -2.0);
    
    // Multiple spheres at different heights for dynamic scene
    sim.add_sphere(Vec3::new(-2.0, 8.0, 0.0), Vec3::ZERO); // High sphere to fall
    sim.add_sphere(Vec3::new(0.0, 6.0, 0.0), Vec3::ZERO);  // Medium sphere
    sim.add_sphere(Vec3::new(2.0, 4.0, 0.0), Vec3::ZERO);  // Lower sphere
    sim.add_sphere(Vec3::new(1.0, 10.0, 0.0), Vec3::ZERO); // Very high sphere
    
    // TODO: Add physics for boxes and cylinders - currently only spheres work
    // sim.add_box(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO);
    // sim.add_cylinder(Vec3::new(1.0, 3.0, 0.0), 0.5, 1.0, Vec3::ZERO);

    let dt = 0.016_f32;

    if enable_render {
        tracing::info!("Starting simulation loop with dt = {}...", dt);
        let mut i = 0;
        let target_fps = 60.0;
        let frame_duration = Duration::from_secs_f32(1.0 / target_fps);
        
        loop {
            let frame_start = Instant::now();
            
            // Use CPU physics for now since GPU version may not handle all object types
            sim.params.dt = dt;
            sim.step_cpu();

            #[cfg(feature = "render")]
            if let Some(r) = renderer.as_mut() {
                r.update_scene(&sim.spheres, &sim.boxes, &sim.cylinders, &sim.planes);
                if !r.render()? {
                    break;
                }
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
            
            // Frame rate limiting
            let frame_time = frame_start.elapsed();
            if frame_time < frame_duration {
                std::thread::sleep(frame_duration - frame_time);
            }
            
            i += 1;
        }
    } else {
        // Headless mode
        let num_steps = 1000;
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
    }

    if !sim.spheres.is_empty() {
        tracing::info!("Final sphere position: {:?}", sim.spheres[0].pos);
    }

    Ok(())
} 