use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn test_pole_behavior_during_physics() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        initial_angle: 0.0, // Start vertical
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Initial state - pole should be standing UP
    let cart_pos_0 = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos_0 = sim.cylinders[cartpole.pole_idx].pos;
    let joint_y_0 = cart_pos_0.y + config.cart_size.y;
    
    println!("=== Initial State ===");
    println!("Cart Y: {}", cart_pos_0.y);
    println!("Joint Y: {}", joint_y_0);
    println!("Pole Y: {}", pole_pos_0.y);
    println!("Pole above joint: {}", pole_pos_0.y > joint_y_0);
    
    // Run one physics step
    sim.step_cpu();
    
    let cart_pos_1 = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos_1 = sim.cylinders[cartpole.pole_idx].pos;
    let joint_y_1 = cart_pos_1.y + config.cart_size.y;
    
    println!("\n=== After 1 Step ===");
    println!("Cart Y: {}", cart_pos_1.y);
    println!("Joint Y: {}", joint_y_1);
    println!("Pole Y: {}", pole_pos_1.y);
    println!("Pole above joint: {}", pole_pos_1.y > joint_y_1);
    println!("Pole moved down by: {}", pole_pos_0.y - pole_pos_1.y);
    
    // The pole center should STILL be above the joint
    assert!(
        pole_pos_1.y > joint_y_1,
        "After physics, pole should still be ABOVE joint! Joint Y: {}, Pole Y: {}",
        joint_y_1, pole_pos_1.y
    );
    
    // Run several more steps
    for i in 2..10 {
        sim.step_cpu();
        let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
        let joint_y = sim.boxes[cartpole.cart_idx].pos.y + config.cart_size.y;
        
        println!("\n=== After {} Steps ===", i);
        println!("Pole Y: {}, Joint Y: {}, Above: {}", 
                 pole_pos.y, joint_y, pole_pos.y > joint_y);
    }
}

#[test] 
fn test_pole_angle_calculation() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        initial_angle: 0.3, // Tilted right
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Check angle calculation
    let angle = cartpole.get_pole_angle(&sim);
    println!("Initial angle set: {}", config.initial_angle);
    println!("Calculated angle: {}", angle);
    
    assert!(
        (angle - config.initial_angle).abs() < 0.01,
        "Angle calculation wrong. Expected: {}, Got: {}",
        config.initial_angle, angle
    );
    
    // The angle function uses atan2(x, y) where:
    // - x is horizontal displacement
    // - y is vertical displacement
    // For a pole tilted right, x should be positive, y should be positive
    
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let joint_pos = cart_pos + Vec3::new(0.0, config.cart_size.y, 0.0);
    
    let pole_vector = pole_pos - joint_pos;
    println!("Pole vector from joint: {:?}", pole_vector);
    println!("Angle from atan2: {}", pole_vector.x.atan2(pole_vector.y));
    
    // For right tilt, x should be positive
    assert!(pole_vector.x > 0.0, "Tilted right pole should have positive X");
    // Pole should still be above joint
    assert!(pole_vector.y > 0.0, "Pole should be above joint (positive Y)");
}