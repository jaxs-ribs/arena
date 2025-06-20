//! Test revolute joint rigidity and proper rotation behavior

use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3, Vec2};

#[test]
fn test_revolute_joint_rigidity() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.2, // Start at slight angle
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("\n=== Revolute Joint Rigidity Test ===");
    
    // Get initial joint positions
    let joint_anchor_a = sim.revolute_joints[cartpole.joint_idx].anchor_a;
    let joint_anchor_b = sim.revolute_joints[cartpole.joint_idx].anchor_b;
    let initial_cart_anchor = sim.boxes[cartpole.cart_idx].pos + joint_anchor_a;
    let initial_pole_anchor = sim.cylinders[cartpole.pole_idx].pos + joint_anchor_b;
    let initial_separation = (initial_cart_anchor - initial_pole_anchor).length();
    
    println!("Initial joint separation: {:.6}m", initial_separation);
    
    let mut max_separation = initial_separation;
    let mut angle_samples = Vec::new();
    
    // Run simulation and track joint integrity
    for i in 0..120 { // 2 seconds
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
        
        // Check joint rigidity
        let cart_anchor = sim.boxes[cartpole.cart_idx].pos + joint_anchor_a;
        let pole_anchor = sim.cylinders[cartpole.pole_idx].pos + joint_anchor_b;
        let separation = (cart_anchor - pole_anchor).length();
        max_separation = max_separation.max(separation);
        
        // Track pole angle
        if i % 20 == 19 {
            let angle = cartpole.get_pole_angle(&sim);
            angle_samples.push(angle);
            let t = (i + 1) as f32 / 60.0;
            println!("t={:.2}s: separation={:.6}m, angle={:.3} rad ({:.1}°)", 
                     t, separation, angle, angle.to_degrees());
        }
    }
    
    println!("Maximum joint separation: {:.6}m", max_separation);
    
    // Validate joint rigidity
    assert!(max_separation < 0.01, 
            "Joint separated too much: {:.6}m (should be < 0.01m)", max_separation);
    
    // Validate pole rotation
    assert!(angle_samples.len() >= 5, "Need angle samples");
    let total_rotation = (angle_samples.last().unwrap() - angle_samples.first().unwrap()).abs();
    println!("Total pole rotation: {:.3} rad ({:.1}°)", total_rotation, total_rotation.to_degrees());
    
    assert!(total_rotation > 0.1, 
            "Pole didn't rotate enough: {:.3} rad", total_rotation);
    
    println!("✓ Joint remains rigid and pole rotates properly");
}

#[test]
fn test_cart_fixed_pole_rotates() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane with high friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.9;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.15, // Start with noticeable angle
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("\n=== Cart Fixed, Pole Rotates Test ===");
    
    let initial_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let initial_pole_angle = cartpole.get_pole_angle(&sim);
    
    println!("Initial cart position: {:?}", initial_cart_pos);
    println!("Initial pole angle: {:.3} rad ({:.1}°)", initial_pole_angle, initial_pole_angle.to_degrees());
    
    // Test 1: No force applied
    for i in 0..60 {
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
        
        if i % 15 == 14 {
            let cart_pos = sim.boxes[cartpole.cart_idx].pos;
            let pole_angle = cartpole.get_pole_angle(&sim);
            let t = (i + 1) as f32 / 60.0;
            
            println!("t={:.2}s: cart_x={:.6}, pole_angle={:.3} rad ({:.1}°)", 
                     t, cart_pos.x, pole_angle, pole_angle.to_degrees());
        }
    }
    
    let final_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let final_pole_angle = cartpole.get_pole_angle(&sim);
    
    // Validate cart stays fixed
    let cart_movement = (final_cart_pos - initial_cart_pos).length();
    println!("Cart movement: {:.6}m", cart_movement);
    assert!(cart_movement < 0.001, 
            "Cart moved too much: {:.6}m", cart_movement);
    
    // Validate pole rotates
    let pole_rotation = (final_pole_angle - initial_pole_angle).abs();
    println!("Pole rotation: {:.3} rad ({:.1}°)", pole_rotation, pole_rotation.to_degrees());
    assert!(pole_rotation > 0.05, 
            "Pole didn't rotate enough: {:.3} rad", pole_rotation);
    
    println!("✓ Cart stays fixed, pole rotates under gravity");
    
    // Test 2: Apply force and verify cart moves
    let pre_force_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    for _ in 0..30 {
        cartpole.apply_force(&mut sim, 1.0); // Apply force
        sim.step_cpu();
    }
    
    let post_force_x = sim.boxes[cartpole.cart_idx].pos.x;
    let cart_displacement = post_force_x - pre_force_x;
    
    println!("Cart displacement with force: {:.3}m", cart_displacement);
    assert!(cart_displacement > 0.1, 
            "Cart didn't respond to force: {:.3}m", cart_displacement);
    
    println!("✓ Cart responds to applied force");
}

#[test]
fn test_joint_anchor_calculation() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.0, // Start vertical for clear testing
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("\n=== Joint Anchor Calculation Test ===");
    
    let joint_anchor_a = sim.revolute_joints[cartpole.joint_idx].anchor_a;
    let joint_anchor_b = sim.revolute_joints[cartpole.joint_idx].anchor_b;
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let pole_half_height = sim.cylinders[cartpole.pole_idx].half_height;
    
    println!("Cart position: {:?}", cart_pos);
    println!("Cart size: {:?}", sim.boxes[cartpole.cart_idx].half_extents);
    println!("Pole position: {:?}", pole_pos);
    println!("Pole half height: {:.3}", pole_half_height);
    
    let cart_anchor_world = cart_pos + joint_anchor_a;
    let pole_anchor_world = pole_pos + joint_anchor_b;
    
    println!("Joint anchor A (cart): {:?}", joint_anchor_a);
    println!("Joint anchor B (pole): {:?}", joint_anchor_b);
    println!("Cart anchor world: {:?}", cart_anchor_world);
    println!("Pole anchor world: {:?}", pole_anchor_world);
    
    let separation = (cart_anchor_world - pole_anchor_world).length();
    println!("Initial anchor separation: {:.6}m", separation);
    
    // Verify the pole anchor is at the bottom of the pole
    let expected_pole_bottom_y = pole_pos.y - pole_half_height;
    let actual_pole_anchor_y = pole_anchor_world.y;
    
    println!("Expected pole bottom Y: {:.3}", expected_pole_bottom_y);
    println!("Actual pole anchor Y: {:.3}", actual_pole_anchor_y);
    
    let y_difference = (expected_pole_bottom_y - actual_pole_anchor_y).abs();
    assert!(y_difference < 0.01, 
            "Pole anchor not at bottom: expected {:.3}, got {:.3}", 
            expected_pole_bottom_y, actual_pole_anchor_y);
    
    // Verify anchors are initially close
    assert!(separation < 0.1, 
            "Initial anchor separation too large: {:.6}m", separation);
    
    println!("✓ Joint anchors calculated correctly");
}