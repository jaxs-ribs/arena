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

use crate::watcher;

// Physics simulation constants
const PHYSICS_TIMESTEP_SECONDS: f32 = 0.016; // 60Hz physics update
const PHYSICS_UPDATE_FREQUENCY: f32 = 60.0;

// Rendering constants
const TARGET_FRAMES_PER_SECOND: f32 = 60.0;

// Logging constants
const FRAMES_BETWEEN_PROGRESS_LOGS: usize = 50;

// Headless mode constants
const HEADLESS_SIMULATION_STEP_COUNT: usize = 1000;

// Scene constants
const GROUND_PLANE_HEIGHT: f32 = 0.0;
const GROUND_PLANE_SIZE: f32 = 15.0; // Creates 30x30 meter area
const GRAVITY_ACCELERATION: f32 = -9.81;

// Test sphere constants
const SPHERE_STARTING_HEIGHT: f32 = 5.0;
const SPHERE_RADIUS: f32 = 0.5;
const SPHERE_SPACING: f32 = 1.5;

/// Run the physics simulation with optional rendering.
///
/// # Arguments
/// * `enable_render` - Whether to display the simulation in a window
///
/// # Errors
/// Returns errors from physics engine, renderer, or file watcher initialization
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

/// Initialize shader file watcher for development hot-reloading.
/// Non-critical: returns None on failure without stopping execution.
fn initialize_shader_watcher() -> Option<notify::RecommendedWatcher> {
    match watcher::start() {
        Ok(watcher_instance) => {
            tracing::info!("Shader watcher started successfully.");
            Some(watcher_instance)
        }
        Err(error) => {
            tracing::error!("Failed to start shader watcher: {error:?}");
            None
        }
    }
}

/// Create demonstration scene with falling spheres.
fn create_test_scene() -> Result<PhysicsSim> {
    tracing::info!("Initializing physics simulation...");
    let mut simulation = PhysicsSim::new();

    configure_world_physics(&mut simulation);
    add_ground_plane(&mut simulation);
    add_falling_spheres(&mut simulation);

    Ok(simulation)
}

/// Configure global physics parameters.
fn configure_world_physics(simulation: &mut PhysicsSim) {
    simulation.params.gravity = create_earth_gravity();
}

fn create_earth_gravity() -> Vec3 {
    Vec3::new(0.0, GRAVITY_ACCELERATION, 0.0)
}

/// Add ground plane for spheres to land on.
fn add_ground_plane(simulation: &mut PhysicsSim) {
    let upward_normal = Vec3::new(0.0, 1.0, 0.0);
    let distance_from_origin = GROUND_PLANE_HEIGHT;
    let visible_area = Vec2::new(GROUND_PLANE_SIZE, GROUND_PLANE_SIZE);
    
    simulation.add_plane(upward_normal, distance_from_origin, visible_area);
}

/// Add three falling spheres to demonstrate physics.
fn add_falling_spheres(simulation: &mut PhysicsSim) {
    let sphere_positions = [
        create_sphere_position(0.0),          // Center
        create_sphere_position(-SPHERE_SPACING), // Left
        create_sphere_position(SPHERE_SPACING),  // Right
    ];
    
    for position in sphere_positions {
        add_sphere_at_position(simulation, position);
    }
}

fn create_sphere_position(x_offset: f32) -> Vec3 {
    Vec3::new(x_offset, SPHERE_STARTING_HEIGHT, 0.0)
}

fn add_sphere_at_position(simulation: &mut PhysicsSim, position: Vec3) {
    let initial_velocity = Vec3::ZERO;
    simulation.add_sphere(position, initial_velocity, SPHERE_RADIUS);
}

/// Run the simulation with rendering enabled
#[cfg(feature = "render")]
fn run_with_rendering(mut sim: PhysicsSim) -> Result<()> {
    let renderer_config = render::RendererConfig::default();
    let (mut renderer, event_loop) = render::Renderer::new(renderer_config)?;
    
    tracing::info!("Starting simulation loop with dt = {}...", PHYSICS_TIMESTEP_SECONDS);
    
    let mut frame_counter = 0;
    let frame_duration = calculate_frame_duration();

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
                
                advance_physics_one_step(&mut sim, frame_counter);
                
                if render_frame(&mut renderer, &sim).is_err() {
                    elwt.exit();
                }
                
                maintain_target_framerate(frame_start, frame_duration);
                
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

/// Run physics simulation without graphical output.
fn run_headless(mut simulation: PhysicsSim) -> Result<()> {
    log_headless_startup();
    
    for frame_number in 0..HEADLESS_SIMULATION_STEP_COUNT {
        if let Err(error) = run_single_physics_step(&mut simulation) {
            log_simulation_error(frame_number, error);
            break;
        }
        
        log_progress_if_interval_reached(&simulation, frame_number);
    }
    
    Ok(())
}

fn log_headless_startup() {
    tracing::info!(
        "Starting headless simulation for {} steps with dt = {}...",
        HEADLESS_SIMULATION_STEP_COUNT,
        PHYSICS_TIMESTEP_SECONDS
    );
}

fn run_single_physics_step(simulation: &mut PhysicsSim) -> Result<(), physics::PhysicsError> {
    simulation.run(PHYSICS_TIMESTEP_SECONDS, 1).map(|_| ())
}

fn log_simulation_error(frame_number: usize, error: physics::PhysicsError) {
    tracing::error!("Error during simulation step {}: {:?}", frame_number, error);
}

/// Advance physics simulation by one timestep.
fn advance_physics_one_step(simulation: &mut PhysicsSim, frame_number: usize) {
    simulation.params.dt = PHYSICS_TIMESTEP_SECONDS;
    simulation.step_cpu();
    
    log_progress_if_interval_reached(simulation, frame_number);
}

/// Update renderer with physics state and render next frame.
#[cfg(feature = "render")]
fn render_frame(renderer: &mut render::Renderer, simulation: &PhysicsSim) -> Result<()> {
    update_renderer_scene_data(renderer, simulation);
    renderer.render(PHYSICS_TIMESTEP_SECONDS)
}

fn update_renderer_scene_data(renderer: &mut render::Renderer, simulation: &PhysicsSim) {
    renderer.update_scene(
        &simulation.spheres,
        &simulation.boxes,
        &simulation.cylinders,
        &simulation.planes
    );
}

/// Log progress every N frames for monitoring.
fn log_progress_if_interval_reached(simulation: &PhysicsSim, frame_number: usize) {
    if !is_progress_log_frame(frame_number) {
        return;
    }
    
    let step_number = frame_number + 1;
    
    if simulation.spheres.is_empty() {
        log_empty_simulation_progress(step_number);
    } else {
        log_sphere_position_progress(step_number, simulation.spheres[0].pos.y);
    }
}

fn is_progress_log_frame(frame_number: usize) -> bool {
    (frame_number + 1) % FRAMES_BETWEEN_PROGRESS_LOGS == 0
}

fn log_empty_simulation_progress(step_number: usize) {
    tracing::info!("Simulation step {} complete. No spheres in simulation.", step_number);
}

fn log_sphere_position_progress(step_number: usize, y_position: f32) {
    tracing::info!(
        "Simulation step {} complete. First sphere Y position: {:.3}",
        step_number,
        y_position
    );
}

/// Sleep to maintain consistent framerate.
fn maintain_target_framerate(frame_start_time: Instant, target_frame_duration: Duration) {
    let elapsed_time = frame_start_time.elapsed();
    
    if frame_completed_too_quickly(elapsed_time, target_frame_duration) {
        sleep_for_remaining_frame_time(elapsed_time, target_frame_duration);
    }
}

fn frame_completed_too_quickly(elapsed: Duration, target: Duration) -> bool {
    elapsed < target
}

fn sleep_for_remaining_frame_time(elapsed: Duration, target: Duration) {
    let sleep_duration = target - elapsed;
    std::thread::sleep(sleep_duration);
}

fn calculate_frame_duration() -> Duration {
    Duration::from_secs_f32(1.0 / TARGET_FRAMES_PER_SECOND)
} 