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
    PhysicsSim, CartPoleGrid, CartPoleConfig,
};
use std::time::{Duration, Instant};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::KeyCode;

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

/// Demo scene type
enum DemoScene {
    FallingSpheres,
    CartPoleGym,
}

// Set the demo scene to use
const ACTIVE_DEMO: DemoScene = DemoScene::CartPoleGym;

/// Create demonstration scene based on active demo type.
fn create_test_scene() -> Result<(PhysicsSim, Option<CartPoleGrid>)> {
    tracing::info!("Initializing physics simulation...");
    let mut simulation = PhysicsSim::new();

    configure_world_physics(&mut simulation);
    
    match ACTIVE_DEMO {
        DemoScene::FallingSpheres => {
            add_ground_plane(&mut simulation);
            add_falling_spheres(&mut simulation);
            Ok((simulation, None))
        }
        DemoScene::CartPoleGym => {
            let grid = create_cartpole_scene(&mut simulation);
            Ok((simulation, Some(grid)))
        }
    }
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

/// Create CartPole gym scene with multiple cartpoles
fn create_cartpole_scene(simulation: &mut PhysicsSim) -> CartPoleGrid {
    tracing::info!("Creating CartPole gym scene...");
    
    // Add ground plane first
    add_ground_plane(simulation);
    
    // Configure cartpoles
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.05,
        pole_mass: 0.1,
        initial_angle: 0.05, // Small random perturbation
        force_magnitude: 10.0,
        failure_angle: 0.5, // ~28 degrees
        position_limit: 3.0,
    };
    
    // Create a 2x3 grid of cartpoles with 2.0 spacing (within 3.0 position limit)
    let grid = CartPoleGrid::new(simulation, 2, 3, 2.0, config);
    
    tracing::info!("Created {} cartpoles in {}x{} grid", 
                  grid.cartpoles.len(), grid.grid_size.0, grid.grid_size.1);
    
    // Log initial physics state
    tracing::info!("Initial physics state:");
    tracing::info!("  - {} boxes (carts)", simulation.boxes.len());
    tracing::info!("  - {} cylinders (poles)", simulation.cylinders.len());
    tracing::info!("  - {} revolute joints", simulation.revolute_joints.len());
    tracing::info!("  - Gravity: ({:.2}, {:.2}, {:.2})", 
                  simulation.params.gravity.x, 
                  simulation.params.gravity.y, 
                  simulation.params.gravity.z);
    
    // Log initial cartpole states
    let states = grid.get_all_states(simulation);
    tracing::info!("Initial CartPole positions:");
    for (i, state) in states.iter().enumerate() {
        tracing::info!("  CartPole {}: x={:.2}, θ={:.3} rad ({:.1}°)", 
                     i, state[0], state[2], state[2].to_degrees());
    }
    
    grid
}

/// Run the simulation with rendering enabled
#[cfg(feature = "render")]
fn run_with_rendering(sim_and_grid: (PhysicsSim, Option<CartPoleGrid>)) -> Result<()> {
    let (mut sim, mut cartpole_grid) = sim_and_grid;
    let renderer_config = render::RendererConfig::default();
    let (mut renderer, event_loop) = render::Renderer::new(renderer_config)?;
    
    tracing::info!("Starting simulation loop with dt = {}...", PHYSICS_TIMESTEP_SECONDS);
    
    if cartpole_grid.is_some() {
        tracing::info!("=== CartPole Demo Controls ===");
        tracing::info!("Press 'M' to toggle manual control mode");
        tracing::info!("When manual control is active:");
        tracing::info!("  - Number keys 1-6: Select cartpole");
        tracing::info!("  - Left/Right arrows: Apply force");
        tracing::info!("  - Space: Stop force");
        tracing::info!("  - R: Reset all cartpoles");
        tracing::info!("Camera controls: WASD to move, mouse to look");
        tracing::info!("Other: P=screenshot, F=fullscreen, ESC=release mouse");
    }
    
    let mut frame_counter = 0;
    let frame_duration = calculate_frame_duration();
    let mut demo_timer = 0.0;
    let mut manual_control = ManualControl::new();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        // Let renderer handle input events and get key presses
        if let Some(keycode) = renderer.handle_event(&event) {
            if let Some(ref mut grid) = cartpole_grid {
                manual_control.handle_key(keycode, grid, &mut sim);
            }
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::AboutToWait => {
                let frame_start = Instant::now();
                
                // Update CartPole demo if active
                if let Some(ref mut grid) = cartpole_grid {
                    if manual_control.is_active() {
                        manual_control.update(grid, &mut sim);
                    } else {
                        update_cartpole_demo(grid, &mut sim, demo_timer);
                    }
                    demo_timer += PHYSICS_TIMESTEP_SECONDS;
                }
                
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
fn run_with_rendering(_sim_and_grid: (PhysicsSim, Option<CartPoleGrid>)) -> Result<()> {
    tracing::error!("Rendering requested but 'render' feature not enabled");
    Ok(())
}

/// Run physics simulation without graphical output.
fn run_headless(sim_and_grid: (PhysicsSim, Option<CartPoleGrid>)) -> Result<()> {
    let (mut simulation, mut cartpole_grid) = sim_and_grid;
    log_headless_startup();
    
    for frame_number in 0..HEADLESS_SIMULATION_STEP_COUNT {
        // Update CartPole demo if active
        if let Some(ref mut grid) = cartpole_grid {
            let time = frame_number as f32 * PHYSICS_TIMESTEP_SECONDS;
            update_cartpole_demo(grid, &mut simulation, time);
        }
        
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

/// Manual control state for CartPoles
struct ManualControl {
    active: bool,
    selected_cartpole: usize,
    action: f32,
}

impl ManualControl {
    fn new() -> Self {
        Self {
            active: false,
            selected_cartpole: 0,
            action: 0.0,
        }
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
    
    fn handle_key(&mut self, keycode: KeyCode, grid: &mut CartPoleGrid, sim: &mut PhysicsSim) {
        match keycode {
            // Toggle manual control
            KeyCode::KeyM => {
                self.active = !self.active;
                self.action = 0.0;
                tracing::info!("Manual control: {}", if self.active { "ON" } else { "OFF" });
                if self.active {
                    tracing::info!("Use number keys 1-6 to select cartpole, arrow keys to control");
                }
            }
            
            // Select cartpole
            KeyCode::Digit1 => self.select_cartpole(0, grid.cartpoles.len()),
            KeyCode::Digit2 => self.select_cartpole(1, grid.cartpoles.len()),
            KeyCode::Digit3 => self.select_cartpole(2, grid.cartpoles.len()),
            KeyCode::Digit4 => self.select_cartpole(3, grid.cartpoles.len()),
            KeyCode::Digit5 => self.select_cartpole(4, grid.cartpoles.len()),
            KeyCode::Digit6 => self.select_cartpole(5, grid.cartpoles.len()),
            
            // Control actions
            KeyCode::ArrowLeft => {
                if self.active {
                    self.action = -1.0;
                    tracing::info!("CartPole {}: Force LEFT", self.selected_cartpole);
                }
            }
            KeyCode::ArrowRight => {
                if self.active {
                    self.action = 1.0;
                    tracing::info!("CartPole {}: Force RIGHT", self.selected_cartpole);
                }
            }
            KeyCode::Space => {
                if self.active {
                    self.action = 0.0;
                    tracing::info!("CartPole {}: Force STOP", self.selected_cartpole);
                }
            }
            
            // Reset all
            KeyCode::KeyR => {
                tracing::info!("Resetting all cartpoles");
                for cartpole in grid.cartpoles.iter_mut() {
                    cartpole.reset(sim);
                }
            }
            
            _ => {}
        }
    }
    
    fn select_cartpole(&mut self, index: usize, max: usize) {
        if self.active && index < max {
            self.selected_cartpole = index;
            tracing::info!("Selected CartPole {}", index);
        }
    }
    
    fn update(&self, grid: &mut CartPoleGrid, sim: &mut PhysicsSim) {
        if !self.active {
            return;
        }
        
        // Apply manual control to selected cartpole
        let mut actions = vec![0.0; grid.cartpoles.len()];
        if self.selected_cartpole < actions.len() {
            actions[self.selected_cartpole] = self.action;
        }
        
        grid.apply_actions(sim, &actions);
        
        // Check and reset failures
        let failed_indices = grid.check_and_reset_failures(sim);
        if !failed_indices.is_empty() {
            tracing::info!("CartPoles failed and reset: {:?}", failed_indices);
        }
        
        // Show selected cartpole state
        if self.selected_cartpole < grid.cartpoles.len() {
            let state = grid.cartpoles[self.selected_cartpole].get_state(sim);
            tracing::debug!("CartPole {}: x={:.2}, v={:.2}, θ={:.1}°, ω={:.2}", 
                          self.selected_cartpole, state[0], state[1], 
                          state[2].to_degrees(), state[3]);
        }
    }
}

/// Update CartPole demo with test actions
fn update_cartpole_demo(grid: &mut CartPoleGrid, sim: &mut PhysicsSim, time: f32) {
    // Create test actions for each cartpole
    let mut actions = Vec::new();
    
    for (i, cartpole) in grid.cartpoles.iter().enumerate() {
        // Different test patterns for each cartpole
        let action = match i {
            0 => (time * 2.0).sin(),           // Oscillating
            1 => if time.sin() > 0.0 { 1.0 } else { -1.0 }, // Bang-bang control
            2 => 0.0,                          // No control (should fall)
            3 => (time * 0.5).cos() * 0.5,    // Slow oscillation
            4 => if cartpole.get_pole_angle(sim) > 0.0 { -0.8 } else { 0.8 }, // Simple feedback
            5 => ((time * 3.0).sin() + (time * 1.5).cos()) * 0.7, // Complex pattern
            _ => 0.0,
        };
        actions.push(action);
    }
    
    // Apply actions
    grid.apply_actions(sim, &actions);
    
    // Check and reset failures
    let failed_indices = grid.check_and_reset_failures(sim);
    
    // Log failures  
    if !failed_indices.is_empty() {
        for idx in &failed_indices {
            // Get state before reset to see why it failed
            let cart_pos = sim.boxes[grid.cartpoles[*idx].cart_idx].pos;
            let pole_angle = grid.cartpoles[*idx].get_pole_angle(sim);
            tracing::info!("CartPole {} failed - x={:.2} (limit: {:.1}), θ={:.3} rad/{:.1}° (limit: {:.1}°)", 
                         idx, cart_pos.x, grid.cartpoles[*idx].config.position_limit,
                         pole_angle, pole_angle.to_degrees(), 
                         grid.cartpoles[*idx].config.failure_angle.to_degrees());
        }
    }
    
    // More frequent status updates for first 10 seconds
    let update_interval = if time < 10.0 { 1000 } else { 5000 };
    if (time * 1000.0) as i32 % update_interval < 16 {
        tracing::info!("=== CartPole Demo Status at t={:.1}s ===", time);
        let states = grid.get_all_states(sim);
        for (i, state) in states.iter().enumerate() {
            let control_desc = match i {
                0 => "Oscillating control",
                1 => "Bang-bang control",
                2 => "No control (falling)",
                3 => "Slow oscillation",
                4 => "Simple feedback",
                5 => "Complex pattern",
                _ => "Unknown",
            };
            tracing::info!("  CartPole {}: {} - x={:.2}, θ={:.1}°, action={:.2}", 
                         i, control_desc, state[0], state[2].to_degrees(), actions[i]);
        }
        
        // Show physics stats
        tracing::info!("Physics stats: {} boxes, {} cylinders, {} revolute joints",
                     sim.boxes.len(), sim.cylinders.len(), sim.revolute_joints.len());
    }
} 