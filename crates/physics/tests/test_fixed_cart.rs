//! Test that carts stay completely fixed without user input

use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3, Vec2};

#[test]
fn test_cart_stays_completely_fixed() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane with high friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.3, // Larger initial angle to create more torque
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let initial_pole_angle = cartpole.get_pole_angle(&sim);
    
    println!("\nInitial state:");
    println!("Cart position: {:?}", initial_cart_pos);
    println!("Pole angle: {:.3} rad ({:.1}°)", initial_pole_angle, initial_pole_angle.to_degrees());
    
    // Run for 60 steps (1 second) with NO applied force
    for i in 0..60 {
        // Apply ZERO force
        cartpole.apply_force(&mut sim, 0.0);
        sim.step_cpu();
        
        if i % 10 == 9 {
            let cart_pos = sim.boxes[cartpole.cart_idx].pos;
            let pole_angle = cartpole.get_pole_angle(&sim);
            let t = (i + 1) as f32 / 60.0;
            
            println!("t={:.2}s: cart_x={:.6}, pole_angle={:.3} rad ({:.1}°)", 
                     t, cart_pos.x, pole_angle, pole_angle.to_degrees());
        }
    }
    
    let final_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let final_pole_angle = cartpole.get_pole_angle(&sim);
    
    println!("\nFinal state:");
    println!("Cart position: {:?}", final_cart_pos);
    println!("Pole angle: {:.3} rad ({:.1}°)", final_pole_angle, final_pole_angle.to_degrees());
    
    // Cart should NOT move AT ALL (within numerical precision)
    let cart_movement = (final_cart_pos - initial_cart_pos).length();
    println!("Cart total movement: {:.6}m", cart_movement);
    
    assert!(cart_movement < 0.001, 
            "Cart moved {:.6}m without applied force - should be completely static!", 
            cart_movement);
    
    // Pole should have fallen (rotated around the cart)
    assert!((final_pole_angle - initial_pole_angle).abs() > 0.1,
            "Pole didn't rotate enough: {:.3} → {:.3} rad",
            initial_pole_angle, final_pole_angle);
}

#[test]
fn test_cart_moves_only_with_applied_force() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane with lower friction for movement
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.3;
    
    let config = CartPoleConfig::default();
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Apply rightward force for 30 steps
    for _ in 0..30 {
        cartpole.apply_force(&mut sim, 1.0); // Apply rightward force
        sim.step_cpu();
    }
    
    let mid_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Stop applying force for 30 steps
    for _ in 0..30 {
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
    }
    
    let final_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    println!("Cart movement test:");
    println!("  Initial: {:.3}", initial_cart_x);
    println!("  With force: {:.3}", mid_cart_x);
    println!("  After stopping force: {:.3}", final_cart_x);
    
    // Cart should have moved when force was applied
    assert!(mid_cart_x > initial_cart_x + 0.1, 
            "Cart didn't move enough with applied force");
    
    // Cart should slow down when force is removed (but may continue moving due to momentum)
    let with_force_velocity = (mid_cart_x - initial_cart_x) / 0.5; // velocity during force phase
    let no_force_velocity = (final_cart_x - mid_cart_x) / 0.5; // velocity after force removed
    
    println!("  Velocity with force: {:.3}", with_force_velocity);
    println!("  Velocity without force: {:.3}", no_force_velocity);
    
    // Velocity should decrease when force is removed
    assert!(no_force_velocity.abs() < with_force_velocity.abs() * 0.8,
            "Cart didn't slow down when force was removed");
}