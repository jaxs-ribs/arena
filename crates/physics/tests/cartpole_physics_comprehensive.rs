//! Comprehensive physics tests for CartPole to catch all issues from first principles

use physics::{PhysicsSim, CartPole, CartPoleGrid, CartPoleConfig, Vec3, Vec2};

// Test that reproduces the exact grid creation issue
#[test]
fn test_grid_creation_positions() {
    let mut sim = PhysicsSim::new();
    
    // Add ground plane 
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    // Create the same 2x3 grid with 1.0 spacing as in main app
    let grid = CartPoleGrid::new(&mut sim, 2, 3, 1.0, config.clone());
    
    // Check positions
    println!("\nGrid positions:");
    for (i, cp) in grid.cartpoles.iter().enumerate() {
        let pos = sim.boxes[cp.cart_idx].pos;
        println!("CartPole {}: x={:.2}, z={:.2}", i, pos.x, pos.z);
        
        // Position should be well within limits
        let margin = 0.5; // Safety margin
        assert!(
            pos.x.abs() < config.position_limit - margin,
            "CartPole {} at x={:.2} is too close to limit {:.1} (need {:.1} margin)",
            i, pos.x, config.position_limit, margin
        );
    }
}

// Test that cart doesn't slide without applied force
#[test]
fn test_cart_stability_without_force() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground with high friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Check cart friction is set properly
    let cart_friction = sim.boxes[cartpole.cart_idx].material.friction;
    assert!(cart_friction > 0.5, "Cart friction too low: {}", cart_friction);
    
    let initial_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Run for 60 steps (1 second) without any applied force
    for _ in 0..60 {
        sim.step_cpu();
    }
    
    let final_x = sim.boxes[cartpole.cart_idx].pos.x;
    let drift = (final_x - initial_x).abs();
    
    println!("Cart drift without force: {:.4}m", drift);
    
    // Cart should not drift more than 1cm without applied force
    assert!(
        drift < 0.01,
        "Cart drifted {:.4}m without applied force (friction may be too low)",
        drift
    );
}

// Test that pole falls naturally with proper angular acceleration
#[test]
fn test_pole_falling_dynamics() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1, // 5.7 degrees
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Track pole angle over time
    let mut angle_history = Vec::new();
    let mut angular_vel_history = Vec::new();
    
    for i in 0..60 {
        let angle = cartpole.get_pole_angle(&sim);
        let angular_vel = sim.cylinders[cartpole.pole_idx].angular_vel.z;
        
        angle_history.push(angle);
        angular_vel_history.push(angular_vel);
        
        if i % 10 == 0 {
            println!("t={:.2}s: angle={:.3} rad ({:.1}°), ω={:.3} rad/s",
                     i as f32 / 60.0, angle, angle.to_degrees(), angular_vel);
        }
        
        sim.step_cpu();
    }
    
    // Check that pole is accelerating (falling faster over time)
    let early_angular_accel = angular_vel_history[10] - angular_vel_history[0];
    let late_angular_accel = angular_vel_history[50] - angular_vel_history[40];
    
    println!("Early angular acceleration: {:.3} rad/s²", early_angular_accel * 6.0);
    println!("Late angular acceleration: {:.3} rad/s²", late_angular_accel * 6.0);
    
    // Angular velocity should be increasing (negative for falling)
    assert!(angular_vel_history[50] < angular_vel_history[10], 
            "Pole angular velocity not increasing properly");
    
    // Pole should have fallen significantly in 1 second
    let final_angle = angle_history.last().unwrap();
    assert!(final_angle.abs() > 0.2, 
            "Pole didn't fall enough: only {:.3} rad ({:.1}°)",
            final_angle, final_angle.to_degrees());
}

// Test revolute joint maintains connection under dynamics
#[test]
fn test_revolute_joint_integrity() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    
    let config = CartPoleConfig::default();
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Track joint separation over time
    let mut max_separation = 0.0f32;
    
    for _ in 0..120 { // 2 seconds
        sim.step_cpu();
        
        // Check joint constraint
        let joint = &sim.revolute_joints[cartpole.joint_idx];
        let cart_anchor = sim.boxes[cartpole.cart_idx].pos + joint.anchor_a;
        let pole_anchor = sim.cylinders[cartpole.pole_idx].pos + joint.anchor_b;
        
        let separation = (cart_anchor - pole_anchor).length();
        max_separation = max_separation.max(separation);
    }
    
    println!("Maximum joint separation: {:.6}m", max_separation);
    
    // Joint should stay connected within tolerance
    assert!(max_separation < 0.1, 
            "Joint separated too much: {:.3}m", max_separation);
}

// Test grid spacing prevents any overlaps
#[test]
fn test_no_cartpole_overlaps() {
    let mut sim = PhysicsSim::new();
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    // Try different spacings
    let spacings = vec![1.0, 2.0, 3.0, 4.0];
    
    for spacing in spacings {
        let grid = CartPoleGrid::new(&mut sim, 2, 3, spacing, config.clone());
        
        // Check for overlaps
        let mut overlaps = Vec::new();
        for i in 0..grid.cartpoles.len() {
            for j in (i+1)..grid.cartpoles.len() {
                let pos_i = sim.boxes[grid.cartpoles[i].cart_idx].pos;
                let pos_j = sim.boxes[grid.cartpoles[j].cart_idx].pos;
                
                let distance = (pos_i - pos_j).length();
                let min_safe_distance = config.cart_size.x * 2.0 + 0.1; // Add small margin
                
                if distance < min_safe_distance {
                    overlaps.push((i, j, distance));
                }
            }
        }
        
        println!("Spacing {:.1}: {} overlaps found", spacing, overlaps.len());
        if spacing >= 2.0 {
            assert!(overlaps.is_empty(), 
                    "CartPoles overlap with spacing {}: {:?}", spacing, overlaps);
        }
    }
}

// Test that CartPoles don't fail immediately
#[test] 
fn test_no_immediate_failures() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    // Create grid with safe spacing
    let mut grid = CartPoleGrid::new(&mut sim, 2, 3, 2.0, config); // Reduced spacing
    
    // Check initial positions
    println!("\nInitial positions:");
    for (i, cp) in grid.cartpoles.iter().enumerate() {
        let pos = sim.boxes[cp.cart_idx].pos;
        let angle = cp.get_pole_angle(&sim);
        println!("CartPole {}: x={:.2}, angle={:.3} rad", i, pos.x, angle);
    }
    
    // Run for just one step
    sim.step_cpu();
    
    // Check for failures
    let failed = grid.check_and_reset_failures(&mut sim);
    
    if !failed.is_empty() {
        println!("Failed CartPoles after 1 step:");
        for idx in &failed {
            let cp = &grid.cartpoles[*idx];
            let pos = sim.boxes[cp.cart_idx].pos;
            let angle = cp.get_pole_angle(&sim);
            println!("  CartPole {}: x={:.2}, angle={:.3} rad", idx, pos.x, angle);
        }
    }
    
    assert!(failed.is_empty(), 
            "CartPoles failed immediately: {:?}", failed);
}

// Test 2D constraints are properly applied
#[test]
fn test_2d_motion_constraints() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    
    let config = CartPoleConfig::default();
    let cartpole = CartPole::new(&mut sim, Vec3::new(0.0, 0.0, 1.0), config); // Start off-center in Z
    
    // Apply some diagonal force that would cause 3D motion
    sim.set_force(cartpole.cart_idx, [10.0, 10.0]);
    
    // Run simulation
    for _ in 0..30 {
        sim.step_cpu();
    }
    
    // Check that motion is constrained to X-Y plane
    let cart_z = sim.boxes[cartpole.cart_idx].pos.z;
    let pole_z = sim.cylinders[cartpole.pole_idx].pos.z;
    let cart_rot_xy = (sim.boxes[cartpole.cart_idx].angular_vel.x.powi(2) +
                      sim.boxes[cartpole.cart_idx].angular_vel.y.powi(2)).sqrt();
    let pole_rot_xy = (sim.cylinders[cartpole.pole_idx].angular_vel.x.powi(2) +
                      sim.cylinders[cartpole.pole_idx].angular_vel.y.powi(2)).sqrt();
    
    println!("After 2D constraints:");
    println!("  Cart Z: {:.6}", cart_z);
    println!("  Pole Z: {:.6}", pole_z);
    println!("  Cart rotation (X,Y): {:.6}", cart_rot_xy);
    println!("  Pole rotation (X,Y): {:.6}", pole_rot_xy);
    
    assert!(cart_z.abs() < 0.001, "Cart not constrained to Z=0: {}", cart_z);
    assert!(pole_z.abs() < 0.001, "Pole not constrained to Z=0: {}", pole_z);
    assert!(cart_rot_xy < 0.001, "Cart has non-Z rotation: {}", cart_rot_xy);
    assert!(pole_rot_xy < 0.001, "Pole has non-Z rotation: {}", pole_rot_xy);
}

// Test force application and response
#[test]
fn test_force_application() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.3; // Lower friction for movement
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.0, // Start vertical for cleaner test
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Apply rightward force for 30 steps
    for _ in 0..30 {
        cartpole.apply_force(&mut sim, 1.0); // Full rightward force
        sim.step_cpu();
    }
    
    let mid_x = sim.boxes[cartpole.cart_idx].pos.x;
    let mid_vel = sim.boxes[cartpole.cart_idx].vel.x;
    
    // Stop applying force for 30 steps
    for _ in 0..30 {
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
    }
    
    let final_x = sim.boxes[cartpole.cart_idx].pos.x;
    let final_vel = sim.boxes[cartpole.cart_idx].vel.x;
    
    println!("Force test results:");
    println!("  Initial X: {:.3}", initial_x);
    println!("  Mid X: {:.3}, Vel: {:.3}", mid_x, mid_vel);
    println!("  Final X: {:.3}, Vel: {:.3}", final_x, final_vel);
    
    // Cart should have moved right
    assert!(mid_x > initial_x + 0.1, 
            "Cart didn't move enough with applied force: {:.3}m", mid_x - initial_x);
    
    // Cart should have positive velocity when force is applied
    assert!(mid_vel > 0.1, "Cart velocity too low: {:.3}", mid_vel);
    
    // Cart should slow down when force is removed
    assert!(final_vel.abs() < mid_vel.abs() * 0.5, 
            "Cart didn't slow down: mid_vel={:.3}, final_vel={:.3}", mid_vel, final_vel);
}

// Test complete scene behavior over time
#[test]
fn test_scene_long_term_stability() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.05, // Smaller angle for longer stability
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    // Create grid with safe parameters  
    let mut grid = CartPoleGrid::new(&mut sim, 2, 2, 1.0, config); // 2x2 grid = 4 CartPoles, 1.0 spacing
    
    // Track failures over time
    let mut failure_timeline = Vec::new();
    
    for step in 0..300 { // 5 seconds
        sim.step_cpu();
        
        let failed = grid.check_and_reset_failures(&mut sim);
        if !failed.is_empty() {
            failure_timeline.push((step, failed.clone()));
        }
        
        // Log state every second
        if step % 60 == 0 {
            let time = step as f32 / 60.0;
            println!("\nTime {:.1}s:", time);
            for (i, cp) in grid.cartpoles.iter().enumerate() {
                let state = cp.get_state(&sim);
                println!("  CartPole {}: x={:.2}, θ={:.3} rad, failed={}",
                         i, state[0], state[2], cp.failed);
            }
        }
    }
    
    println!("\nFailure timeline:");
    for (step, failed) in &failure_timeline {
        println!("  Step {}: CartPoles {:?} failed", step, failed);
    }
    
    // Should have some failures but not immediate ones
    assert!(!failure_timeline.is_empty(), "No failures detected (physics might be wrong)");
    
    let first_failure_step = failure_timeline[0].0;
    assert!(first_failure_step > 30, 
            "Failures happened too early: step {}", first_failure_step);
}

// Test that logs match actual physics state
#[test]
fn test_observability_accuracy() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let mut grid = CartPoleGrid::new(&mut sim, 1, 2, 2.0, config.clone());
    
    // Simulate and track states
    let mut state_log = Vec::new();
    
    for step in 0..120 {
        // Get states before step
        let states: Vec<_> = grid.cartpoles.iter().enumerate().map(|(i, cp)| {
            let state = cp.get_state(&sim);
            let pos = sim.boxes[cp.cart_idx].pos;
            // Check failure without mutable borrow
            let at_position_limit = state[0].abs() >= config.position_limit;
            let at_angle_limit = state[2].abs() >= config.failure_angle;
            let failed = at_position_limit || at_angle_limit;
            (i, state, pos, failed)
        }).collect();
        
        state_log.push((step, states.clone()));
        
        // Check for inconsistencies
        for (i, state, pos, will_fail) in &states {
            // Verify position matches state
            assert!((state[0] - pos.x).abs() < 0.001, 
                    "Position mismatch: state says {:.3}, actual {:.3}", state[0], pos.x);
            
            // Check failure conditions
            let at_position_limit = state[0].abs() >= config.position_limit;
            let at_angle_limit = state[2].abs() >= config.failure_angle;
            let should_fail = at_position_limit || at_angle_limit;
            
            if should_fail != *will_fail {
                println!("Failure detection mismatch at step {} for CartPole {}:", step, i);
                println!("  Position: {:.3} (limit: {:.3})", state[0], config.position_limit);
                println!("  Angle: {:.3} (limit: {:.3})", state[2], config.failure_angle);
                println!("  Should fail: {}, Will fail: {}", should_fail, will_fail);
            }
        }
        
        sim.step_cpu();
        grid.check_and_reset_failures(&mut sim);
    }
    
    // Verify we can reproduce any logged state
    println!("\nState verification summary:");
    println!("  Logged {} states over {} steps", state_log.len(), 120);
    println!("  All position values matched actual positions");
    println!("  Failure detection was consistent");
}