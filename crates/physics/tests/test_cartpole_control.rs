use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::{Vec3, Vec2};

#[test]
fn test_kinematic_cart_velocity_control() {
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig::default();
    
    // Create a cartpole at origin
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Initial position should be at origin (plus cart height)
    let initial_x = sim.boxes[cartpole.cart_idx].pos.x;
    assert_eq!(initial_x, 0.0);
    
    // Apply full right force
    cartpole.apply_force(&mut sim, 1.0);
    
    // Cart velocity should be set immediately (kinematic control)
    let vel_after_force = sim.boxes[cartpole.cart_idx].vel.x;
    assert!(vel_after_force > 0.0, "Cart should have positive velocity");
    
    // Step physics
    sim.step_cpu();
    
    // Cart should have moved right
    let pos_after_step = sim.boxes[cartpole.cart_idx].pos.x;
    assert!(pos_after_step > initial_x, "Cart should have moved right");
    
    // Apply zero force (stop)
    cartpole.apply_force(&mut sim, 0.0);
    
    // Velocity should decelerate smoothly
    let vel_after_stop = sim.boxes[cartpole.cart_idx].vel.x;
    assert!(vel_after_stop < vel_after_force, "Cart should be decelerating");
    
    // After several steps, cart should stop
    for _ in 0..20 {
        cartpole.apply_force(&mut sim, 0.0);
        sim.step_cpu();
    }
    
    let final_vel = sim.boxes[cartpole.cart_idx].vel.x;
    assert!(final_vel.abs() < 0.01, "Cart should have stopped");
}

#[test]
fn test_cart_pole_coupling() {
    let mut sim = PhysicsSim::new();
    
    // Add ground plane so cart can rest
    sim.add_plane(Vec3::new(0.0, -1.0, 0.0), 1.0, Vec2::new(100.0, 100.0));
    
    let config = CartPoleConfig {
        initial_angle: 0.1, // Start with small angle
        ..Default::default()
    };
    
    // Create a cartpole
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Initial pole angle
    let initial_angle = cartpole.get_pole_angle(&sim);
    assert!((initial_angle - 0.1).abs() < 0.01, "Initial angle should be ~0.1 rad");
    
    // Move cart right
    cartpole.apply_force(&mut sim, 1.0);
    sim.step_cpu();
    
    // Cart should have moved
    let cart_pos = sim.boxes[cartpole.cart_idx].pos.x;
    println!("Cart position after step: {}", cart_pos);
    assert!(cart_pos > 0.0, "Cart should have moved right");
    
    // Pole should be affected by cart movement
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    println!("Pole position after step: {:?}", pole_pos);
    println!("Pole angle: {}", cartpole.get_pole_angle(&sim));
    assert!(pole_pos.x > -0.1, "Pole should have moved with cart (allowing for tilt)");
}