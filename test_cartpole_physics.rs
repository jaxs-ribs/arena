#!/usr/bin/env rust-script
//! Test CartPole physics without rendering
//! Run with: cargo run --bin test_cartpole_physics

use physics::{PhysicsSim, CartPoleGrid, CartPoleConfig, Vec3};

fn main() {
    println!("=== CartPole Physics Test ===");
    
    // Create simulation
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.params.dt = 0.016;
    
    // Configure cartpoles
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.05,
        pole_mass: 0.1,
        initial_angle: 0.05, // Small perturbation
        force_magnitude: 10.0,
        failure_angle: 0.5, // ~28 degrees
        position_limit: 3.0,
    };
    
    // Create a single cartpole at origin
    let mut grid = CartPoleGrid::new(&mut sim, 1, 1, 2.0, config);
    
    println!("Initial state:");
    let state = grid.cartpoles[0].get_state(&sim);
    println!("  Position: {:.3}, Velocity: {:.3}", state[0], state[1]);
    println!("  Angle: {:.3} rad ({:.1}°), Angular vel: {:.3}", state[2], state[2].to_degrees(), state[3]);
    
    // Simulate pole falling (no control)
    println!("\nSimulating pole falling (no control):");
    for i in 0..50 {
        sim.step_cpu();
        
        if i % 10 == 9 {
            let state = grid.cartpoles[0].get_state(&sim);
            println!("  Step {}: angle = {:.3} rad ({:.1}°)", i+1, state[2], state[2].to_degrees());
        }
    }
    
    // Check if it failed
    let failed = grid.cartpoles[0].check_failure(&sim);
    println!("\nPole failed: {}", failed);
    
    // Reset and test with control
    grid.cartpoles[0].reset(&mut sim);
    println!("\nReset. Testing with alternating control:");
    
    for i in 0..100 {
        // Simple bang-bang control
        let state = grid.cartpoles[0].get_state(&sim);
        let action = if state[2] > 0.0 { -1.0 } else { 1.0 };
        
        grid.apply_actions(&mut sim, &vec![action]);
        sim.step_cpu();
        
        if i % 20 == 19 {
            let state = grid.cartpoles[0].get_state(&sim);
            println!("  Step {}: x={:.2}, angle={:.3} rad ({:.1}°)", 
                     i+1, state[0], state[2], state[2].to_degrees());
        }
    }
    
    // Final check
    let failed = grid.cartpoles[0].check_failure(&sim);
    println!("\nPole failed with control: {}", failed);
    
    // Test forces
    println!("\n=== Testing Force Application ===");
    grid.cartpoles[0].reset(&mut sim);
    
    // Apply rightward force
    grid.cartpoles[0].apply_force(&mut sim, 1.0);
    for _ in 0..20 {
        sim.step_cpu();
    }
    
    let cart_pos = sim.boxes[grid.cartpoles[0].cart_idx].pos;
    let cart_vel = sim.boxes[grid.cartpoles[0].cart_idx].vel;
    println!("After rightward force: pos.x = {:.3}, vel.x = {:.3}", cart_pos.x, cart_vel.x);
    
    // Test cylinder physics
    println!("\n=== Checking Cylinder Physics ===");
    let pole_idx = grid.cartpoles[0].pole_idx;
    let pole = &sim.cylinders[pole_idx];
    println!("Pole position: ({:.3}, {:.3}, {:.3})", pole.pos.x, pole.pos.y, pole.pos.z);
    println!("Pole orientation: {:?}", pole.orientation);
    println!("Pole angular velocity: ({:.3}, {:.3}, {:.3})", 
             pole.angular_vel.x, pole.angular_vel.y, pole.angular_vel.z);
    
    println!("\n=== Test Complete ===");
}