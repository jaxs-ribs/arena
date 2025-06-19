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
use physics::{
    types::{Vec2, Vec3},
    PhysicsSim,
};
use std::time::{Duration, Instant};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

// render module used in conditional compilation below

use crate::watcher;

// Simulation constants
const SIMULATION_TIMESTEP: f32 = 0.016; // 60Hz physics update
const TARGET_FPS: f32 = 60.0;
const PROGRESS_LOG_INTERVAL: usize = 50;
const HEADLESS_SIMULATION_STEPS: usize = 1000;

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
    initialize_logging();
    
    // Set up shader hot reloading (optional - continues if fails)
    let _shader_watcher = initialize_shader_watcher();

    // Create and configure the physics simulation
    let sim = create_test_scene()?;

    if enable_render {
        run_with_rendering(sim)
    } else {
        run_headless(sim)
    }
}

/// Initialize the tracing subscriber for logging
fn initialize_logging() {
    tracing_subscriber::fmt::init();
}

/// Initialize the shader file watcher for hot reloading
/// Returns None if initialization fails (non-critical)
fn initialize_shader_watcher() -> Option<notify::RecommendedWatcher> {
    match watcher::start() {
        Ok(watcher_instance) => {
            tracing::info!("Shader watcher started successfully.");
            Some(watcher_instance)
        }
        Err(e) => {
            tracing::error!("Failed to start shader watcher: {e:?}");
            None
        }
    }
}

/// Create a test physics scene with various objects
fn create_test_scene() -> Result<PhysicsSim> {
    tracing::info!("Initializing physics simulation...");
    let mut sim = PhysicsSim::new();

    // Configure world physics
    configure_world_physics(&mut sim);
    
    // Add static environment
    add_static_environment(&mut sim);
    
    // Add dynamic test objects
    add_test_objects(&mut sim);

    Ok(sim)
}

/// Configure global physics parameters
fn configure_world_physics(sim: &mut PhysicsSim) {
    // Enable gravity for dynamic motion
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
}

/// Add static environment objects (planes, walls, etc)
fn add_static_environment(sim: &mut PhysicsSim) {
    // Ground plane at y=0
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(25.0, 25.0));

    // Tilted ramp plane for testing rolling/sliding
    let ramp_normal = Vec3::new(0.3, 1.0, 0.0).normalize();
    sim.add_plane(ramp_normal, -2.0, Vec2::new(0.0, 0.0));
}

/// Add dynamic test objects to the scene
fn add_test_objects(sim: &mut PhysicsSim) {
    // Test sphere-sphere collisions with stacked spheres
    add_stacked_spheres(sim);
    
    // Test lateral sphere collisions
    add_collision_test_spheres(sim);
    
    // Add other object types for visual variety
    add_misc_objects(sim);
}

/// Add stacked spheres to test resting contact
fn add_stacked_spheres(sim: &mut PhysicsSim) {
    sim.add_sphere(Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO, 1.0); // Bottom sphere on ground
    sim.add_sphere(Vec3::new(0.0, 5.5, 0.0), Vec3::ZERO, 1.0); // Middle sphere (should rest on bottom)
    sim.add_sphere(Vec3::new(0.0, 8.0, 0.0), Vec3::ZERO, 1.0); // Top sphere (should fall and bounce)
}

/// Add spheres for lateral collision testing
fn add_collision_test_spheres(sim: &mut PhysicsSim) {
    sim.add_sphere(Vec3::new(-3.0, 8.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0); // Moving sphere
    sim.add_sphere(Vec3::new(3.0, 4.0, 0.0), Vec3::ZERO, 1.0); // Stationary target
}

/// Add miscellaneous objects (currently no collision detection for these)
fn add_misc_objects(sim: &mut PhysicsSim) {
    sim.add_box(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO);
    sim.add_cylinder(Vec3::new(5.0, 3.0, 0.0), 0.5, 1.0, Vec3::ZERO);
}

/// Run the simulation with rendering enabled
#[cfg(feature = "render")]
fn run_with_rendering(mut sim: PhysicsSim) -> Result<()> {
    let renderer_config = render::RendererConfig::default();
    let (mut renderer, event_loop) = render::Renderer::new(renderer_config)?;
    
    tracing::info!("Starting simulation loop with dt = {}...", SIMULATION_TIMESTEP);
    
    let mut frame_counter = 0;
    let frame_duration = Duration::from_secs_f32(1.0 / TARGET_FPS);

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        // Let renderer handle input events
        renderer.handle_event(&event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::AboutToWait => {
                let frame_start = Instant::now();
                
                // Perform one physics simulation step
                advance_simulation(&mut sim, frame_counter);
                
                // Update and render the scene
                if update_and_render(&mut renderer, &sim).is_err() {
                    elwt.exit();
                }
                
                // Maintain target framerate
                limit_framerate(frame_start, frame_duration);
                
                frame_counter += 1;
            }
            _ => (),
        }
    })?;
    Ok(())
}

/// Run the simulation in headless mode (no rendering)
#[cfg(not(feature = "render"))]
fn run_with_rendering(_sim: PhysicsSim) -> Result<()> {
    tracing::error!("Rendering requested but 'render' feature not enabled");
    Ok(())
}

/// Run the simulation without rendering
fn run_headless(mut sim: PhysicsSim) -> Result<()> {
    tracing::info!(
        "Starting headless simulation for {} steps with dt = {}...",
        HEADLESS_SIMULATION_STEPS,
        SIMULATION_TIMESTEP
    );
    
    for frame in 0..HEADLESS_SIMULATION_STEPS {
        if let Err(e) = sim.run(SIMULATION_TIMESTEP, 1) {
            tracing::error!("Error during simulation step {}: {:?}", frame, e);
            break;
        }
        
        log_simulation_progress(&sim, frame);
    }
    
    Ok(())
}

/// Advance the physics simulation by one timestep
fn advance_simulation(sim: &mut PhysicsSim, frame: usize) {
    // Use CPU physics for now since GPU version may not handle all object types
    sim.params.dt = SIMULATION_TIMESTEP;
    sim.step_cpu();
    
    log_simulation_progress(sim, frame);
}

/// Update renderer with current scene state and render frame
#[cfg(feature = "render")]
fn update_and_render(renderer: &mut render::Renderer, sim: &PhysicsSim) -> Result<()> {
    renderer.update_scene(&sim.spheres, &sim.boxes, &sim.cylinders, &sim.planes);
    renderer.render(SIMULATION_TIMESTEP)
}

/// Log simulation progress at regular intervals
fn log_simulation_progress(sim: &PhysicsSim, frame: usize) {
    if (frame + 1) % PROGRESS_LOG_INTERVAL == 0 {
        if !sim.spheres.is_empty() {
            tracing::info!(
                "Simulation step {} complete. First sphere Y position: {:.3}",
                frame + 1,
                sim.spheres[0].pos.y
            );
        } else {
            tracing::info!(
                "Simulation step {} complete. No spheres in simulation.",
                frame + 1
            );
        }
    }
}

/// Sleep to maintain target framerate if frame completed early
fn limit_framerate(frame_start: Instant, target_duration: Duration) {
    let frame_time = frame_start.elapsed();
    if frame_time < target_duration {
        std::thread::sleep(target_duration - frame_time);
    }
} 