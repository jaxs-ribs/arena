//! Integration tests for CartPole scene to catch issues before runtime

use physics::{PhysicsSim, CartPoleGrid, CartPoleConfig, Vec3, Vec2};

#[test]
fn test_cartpole_grid_positions_within_limits() {
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
    
    // Create the same grid as in main app with corrected spacing
    let grid = CartPoleGrid::new(&mut sim, 2, 3, 1.0, config.clone());
    
    // Check all CartPole positions
    for (i, cartpole) in grid.cartpoles.iter().enumerate() {
        let cart_pos = sim.boxes[cartpole.cart_idx].pos;
        println!("CartPole {}: x={:.2}, z={:.2}", i, cart_pos.x, cart_pos.z);
        
        // Position should be well within limits (not AT the limit)
        assert!(cart_pos.x.abs() < config.position_limit - 0.1, 
                "CartPole {} at x={:.2} is too close to limit {:.1}", 
                i, cart_pos.x, config.position_limit);
    }
}

#[test] 
fn test_cartpole_doesnt_fail_immediately() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
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
    
    // Create grid with corrected spacing
    let mut grid = CartPoleGrid::new(&mut sim, 2, 3, 1.0, config);
    
    // Run one step
    sim.step_cpu();
    
    // Check for immediate failures
    let failed = grid.check_and_reset_failures(&mut sim);
    assert!(failed.is_empty(), 
            "CartPoles failed immediately on first step: {:?}", failed);
}

#[test]
fn test_pole_angle_changes_over_time() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane
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
    
    // Single CartPole at safe position
    let cartpole = physics::CartPole::new(&mut sim, Vec3::new(0.0, 0.0, 0.0), config);
    
    let initial_angle = cartpole.get_pole_angle(&sim);
    
    // Run for 30 steps (0.5 seconds)
    for _ in 0..30 {
        sim.step_cpu();
    }
    
    let final_angle = cartpole.get_pole_angle(&sim);
    
    println!("Pole angle: initial={:.3}, final={:.3}", initial_angle, final_angle);
    
    // Angle should have changed
    assert!((final_angle - initial_angle).abs() > 0.01,
            "Pole angle didn't change: initial={:.3}, final={:.3}",
            initial_angle, final_angle);
}

#[test]
fn test_cart_position_updates_correctly() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    sim.planes[plane_idx].material.friction = 0.3; // Lower friction for this test
    
    let config = CartPoleConfig::default();
    
    // CartPole at origin
    let cartpole = physics::CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Apply force for 10 steps
    for _ in 0..10 {
        cartpole.apply_force(&mut sim, 1.0);
        sim.step_cpu();
    }
    
    let final_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    println!("Cart position: initial={:.3}, final={:.3}", initial_x, final_x);
    
    // Cart should have moved right
    assert!(final_x > initial_x + 0.01,
            "Cart didn't move right with applied force: moved {:.3}m",
            final_x - initial_x);
}

#[test]
fn test_grid_spacing_prevents_overlap() {
    let mut sim = PhysicsSim::new();
    
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
    
    // Create grid with corrected spacing
    let spacing = 1.0;
    let grid = CartPoleGrid::new(&mut sim, 2, 3, spacing, config.clone());
    
    // Check for overlaps
    for i in 0..grid.cartpoles.len() {
        for j in (i+1)..grid.cartpoles.len() {
            let pos_i = sim.boxes[grid.cartpoles[i].cart_idx].pos;
            let pos_j = sim.boxes[grid.cartpoles[j].cart_idx].pos;
            
            let distance = (pos_i - pos_j).length();
            let min_distance = config.cart_size.x * 2.0; // Double the half-width
            
            assert!(distance > min_distance,
                    "CartPoles {} and {} overlap: distance={:.3}, min={:.3}",
                    i, j, distance, min_distance);
        }
    }
}

#[test]
fn test_revolute_joint_is_functioning() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    
    let config = CartPoleConfig::default();
    let cartpole = physics::CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Get initial joint anchor positions
    let joint = &sim.revolute_joints[cartpole.joint_idx];
    let cart_anchor = sim.boxes[cartpole.cart_idx].pos + joint.anchor_a;
    let pole_anchor = sim.cylinders[cartpole.pole_idx].pos + joint.anchor_b;
    
    let initial_separation = (cart_anchor - pole_anchor).length();
    
    // Run simulation
    for _ in 0..60 {
        sim.step_cpu();
    }
    
    // Check joint is still connected
    let joint = &sim.revolute_joints[cartpole.joint_idx];
    let cart_anchor = sim.boxes[cartpole.cart_idx].pos + joint.anchor_a;
    let pole_anchor = sim.cylinders[cartpole.pole_idx].pos + joint.anchor_b;
    
    let final_separation = (cart_anchor - pole_anchor).length();
    
    println!("Joint separation: initial={:.6}, final={:.6}", 
             initial_separation, final_separation);
    
    // Joint should maintain connection (within tolerance)
    assert!(final_separation < 0.1,
            "Revolute joint separated too much: {:.3}m", final_separation);
}