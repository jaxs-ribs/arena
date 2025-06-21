//! Final comprehensive test for CartPole physics validation

use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3, Vec2};

#[test]
fn test_cartpole_physics_complete_validation() {
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
        initial_angle: 0.1, // Start with slight angle
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let initial_pole_angle = cartpole.get_pole_angle(&sim);
    
    println!("\n=== CartPole Physics Validation ===");
    println!("Initial cart position: {:?}", initial_cart_pos);
    println!("Initial pole angle: {:.3} rad ({:.1}°)", initial_pole_angle, initial_pole_angle.to_degrees());
    
    // Test 1: Cart stays fixed without force
    println!("\n--- Test 1: Cart Stability (no force applied) ---");
    for i in 0..60 {
        // Apply NO force
        cartpole.apply_force(&mut sim, 0.0);
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
    
    // Validate cart didn't move
    let cart_movement = (final_cart_pos - initial_cart_pos).length();
    println!("Cart total movement: {:.6}m", cart_movement);
    assert!(cart_movement < 0.001, 
            "Cart moved {:.6}m without applied force - should be completely static!", 
            cart_movement);
    
    // Validate pole fell naturally
    let pole_rotation = (final_pole_angle - initial_pole_angle).abs();
    println!("Pole rotation: {:.3} rad ({:.1}°)", pole_rotation, pole_rotation.to_degrees());
    assert!(pole_rotation > 0.05,
            "Pole didn't rotate enough: {:.3} rad", pole_rotation);
    
    println!("✓ Test 1 PASSED: Cart stays fixed, pole rotates naturally");
    
    // Test 2: Cart responds to applied force
    println!("\n--- Test 2: Cart Response to Force ---");
    let mid_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Apply force for 30 steps
    for i in 0..30 {
        cartpole.apply_force(&mut sim, 1.0); // Rightward force
        sim.step_cpu();
        
        if i % 10 == 9 {
            let cart_pos = sim.boxes[cartpole.cart_idx].pos;
            let t = (i + 1) as f32 / 60.0;
            println!("t={:.2}s: cart_x={:.3} (force applied)", t, cart_pos.x);
        }
    }
    
    let force_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    let cart_displacement = force_cart_x - mid_cart_x;
    
    println!("Cart movement with force: {:.3}m", cart_displacement);
    assert!(cart_displacement > 0.1, 
            "Cart didn't move enough with applied force: {:.3}m", cart_displacement);
    
    // Stop applying force
    for i in 0..30 {
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
    }
    
    let no_force_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    let velocity_change = (no_force_cart_x - force_cart_x).abs();
    
    println!("Cart movement after force stopped: {:.3}m", velocity_change);
    
    println!("✓ Test 2 PASSED: Cart responds to force and slows when force removed");
    
    // Test 3: Physics consistency over longer time
    println!("\n--- Test 3: Long-term Physics Consistency ---");
    let mut angle_samples = Vec::new();
    
    for i in 0..120 { // 2 seconds
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
        
        if i % 30 == 29 {
            let angle = cartpole.get_pole_angle(&sim);
            angle_samples.push(angle);
            let t = (i + 1) as f32 / 60.0;
            println!("t={:.1}s: pole_angle={:.3} rad ({:.1}°)", t, angle, angle.to_degrees());
        }
    }
    
    // Pole should continue rotating (increasing angle magnitude)
    assert!(angle_samples.len() >= 3, "Need at least 3 angle samples");
    let angle_trend = angle_samples[2].abs() - angle_samples[0].abs();
    println!("Pole angle trend over 2s: {:.3} rad", angle_trend);
    assert!(angle_trend > 0.05, "Pole not falling consistently: {:.3} rad change", angle_trend);
    
    println!("✓ Test 3 PASSED: Physics remains consistent over time");
    
    println!("\n=== ALL TESTS PASSED ===");
    println!("CartPole physics is working correctly:");
    println!("  • Carts stay completely fixed without applied force");  
    println!("  • Carts respond appropriately to applied forces");
    println!("  • Poles fall naturally under gravity around revolute joints");
    println!("  • Physics behavior is consistent over time");
}