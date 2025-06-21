//! Tests for the CartPole entity wrapper

use physics::{
    PhysicsSim, CartPole, CartPoleConfig, CartPoleGrid,
    Vec3,
};

#[test]
fn test_cartpole_creation() {
    println!("\n=== Testing CartPole Creation ===");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig::default();
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Verify components were created
    assert!(cartpole.cart_idx < sim.boxes.len());
    assert!(cartpole.pole_idx < sim.cylinders.len());
    assert!(cartpole.joint_idx < sim.revolute_joints.len());
    
    println!("✓ CartPole created successfully!");
}

#[test]
fn test_cartpole_force_application() {
    println!("\n=== Testing CartPole Force Application ===");
    
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::ZERO; // Disable gravity for this test
    
    let config = CartPoleConfig {
        force_magnitude: 10.0,
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Apply force to the right
    cartpole.apply_force(&mut sim, 1.0);
    
    // Run simulation
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    // Cart should have moved to the right
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    assert!(cart_pos.x > 0.0, "Cart should have moved right, but is at x={}", cart_pos.x);
    
    println!("✓ Force application works!");
}

#[test]
fn test_cartpole_failure_detection() {
    println!("\n=== Testing CartPole Failure Detection ===");
    
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        initial_angle: 0.4, // Start with large angle
        failure_angle: 0.3, // Fail at smaller angle
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Should fail immediately due to angle
    let failed = cartpole.check_failure(&sim);
    assert!(failed, "CartPole should have failed due to large initial angle");
    assert!(cartpole.failed, "Failed flag should be set");
    
    println!("✓ Failure detection works!");
}

#[test]
fn test_cartpole_reset() {
    println!("\n=== Testing CartPole Reset ===");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig::default();
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Apply force and run simulation
    cartpole.apply_force(&mut sim, 1.0);
    for _ in 0..20 {
        sim.step_cpu();
    }
    
    // Cart should have moved
    let cart_pos_before = sim.boxes[cartpole.cart_idx].pos;
    assert!(cart_pos_before.x != 0.0, "Cart should have moved");
    
    // Reset
    cartpole.reset(&mut sim);
    
    // Check reset state
    let cart = &sim.boxes[cartpole.cart_idx];
    assert_eq!(cart.vel, Vec3::ZERO, "Cart velocity should be zero after reset");
    assert!(!cartpole.failed, "Failed flag should be cleared");
    
    // Pole should be at initial angle
    let pole_angle = cartpole.get_pole_angle(&sim);
    assert!((pole_angle - config.initial_angle).abs() < 0.01, 
            "Pole angle should be reset to initial value");
    
    println!("✓ Reset works!");
}

#[test]
fn test_cartpole_state_vector() {
    println!("\n=== Testing CartPole State Vector ===");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig::default();
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let state = cartpole.get_state(&sim);
    
    assert_eq!(state.len(), 4, "State vector should have 4 elements");
    println!("State: cart_x={:.3}, cart_vel={:.3}, pole_angle={:.3}, pole_angular_vel={:.3}",
             state[0], state[1], state[2], state[3]);
    
    println!("✓ State vector works!");
}

#[test]
fn test_cartpole_grid() {
    println!("\n=== Testing CartPole Grid ===");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig::default();
    
    let grid = CartPoleGrid::new(&mut sim, 2, 3, 3.0, config);
    
    // Should have 6 cartpoles (2x3 grid)
    assert_eq!(grid.cartpoles.len(), 6);
    assert_eq!(grid.grid_size, (2, 3));
    
    // Check positioning
    let states = grid.get_all_states(&sim);
    assert_eq!(states.len(), 6);
    
    // First and last cartpole should be separated
    let first_x = states[0][0];
    let last_x = states[5][0];
    assert!((last_x - first_x).abs() > 2.0, "Grid cartpoles should be spaced apart");
    
    println!("✓ CartPole grid works!");
}

#[test]
fn test_grid_reset_failures() {
    println!("\n=== Testing Grid Failure Reset ===");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: 0.05,
        failure_angle: 0.1, // Low failure threshold
        ..Default::default()
    };
    
    let mut grid = CartPoleGrid::new(&mut sim, 1, 3, 3.0, config);
    
    // Run simulation - some should fail
    for _ in 0..50 {
        sim.step_cpu();
    }
    
    // Check and reset failures
    let failed_indices = grid.check_and_reset_failures(&mut sim);
    println!("Failed cartpoles: {:?}", failed_indices);
    
    // Verify reset
    for idx in &failed_indices {
        let state = grid.cartpoles[*idx].get_state(&sim);
        assert!(state[2].abs() < 0.1, "Failed cartpole should be reset to small angle");
    }
    
    println!("✓ Grid failure reset works!");
}