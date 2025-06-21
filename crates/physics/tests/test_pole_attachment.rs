use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn test_pole_stands_up_from_cart() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        initial_angle: 0.0, // Perfectly vertical
        ..Default::default()
    };
    
    // Create cartpole at origin
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Get positions
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    
    // Cart should be at y = cart_half_height
    assert_eq!(cart_pos.y, config.cart_size.y, "Cart should be at correct height");
    
    // Joint is at top of cart
    let joint_y = cart_pos.y + config.cart_size.y;
    
    // Pole center should be ABOVE the joint (standing up)
    // For a vertical pole, center is at joint_y + pole_half_length
    let expected_pole_y = joint_y + config.pole_length / 2.0;
    
    println!("Cart position: {:?}", cart_pos);
    println!("Joint position Y: {}", joint_y);
    println!("Pole position: {:?}", pole_pos);
    println!("Expected pole Y: {}", expected_pole_y);
    
    assert!(
        pole_pos.y > joint_y, 
        "Pole center should be ABOVE joint. Joint Y: {}, Pole Y: {}", 
        joint_y, pole_pos.y
    );
    
    assert!(
        (pole_pos.y - expected_pole_y).abs() < 0.01,
        "Pole should be standing UP. Expected Y: {}, Actual Y: {}",
        expected_pole_y, pole_pos.y
    );
}

#[test]
fn test_tilted_pole_position() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        initial_angle: 0.5, // Tilted to the right
        ..Default::default()
    };
    
    // Create cartpole
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Get positions
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    
    // Joint is at top of cart
    let joint_y = cart_pos.y + config.cart_size.y;
    let joint_x = cart_pos.x;
    
    // For tilted pole, calculate expected position
    let pole_half_length = config.pole_length / 2.0;
    let expected_x = joint_x + pole_half_length * config.initial_angle.sin();
    let expected_y = joint_y + pole_half_length * config.initial_angle.cos();
    
    println!("Initial angle: {} rad", config.initial_angle);
    println!("Joint position: ({}, {})", joint_x, joint_y);
    println!("Pole position: {:?}", pole_pos);
    println!("Expected pole position: ({}, {})", expected_x, expected_y);
    
    // Pole should still be mostly above the joint
    assert!(
        pole_pos.y > joint_y,
        "Even tilted, pole center should be above joint. Joint Y: {}, Pole Y: {}",
        joint_y, pole_pos.y
    );
    
    // Check position matches expected
    assert!(
        (pole_pos.x - expected_x).abs() < 0.01,
        "Pole X incorrect. Expected: {}, Actual: {}",
        expected_x, pole_pos.x
    );
    
    assert!(
        (pole_pos.y - expected_y).abs() < 0.01,
        "Pole Y incorrect. Expected: {}, Actual: {}",
        expected_y, pole_pos.y
    );
}

#[test]
fn test_pole_bottom_at_joint() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        pole_radius: 0.05,
        initial_angle: 0.0,
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let pole_half_height = sim.cylinders[cartpole.pole_idx].half_height;
    
    // Joint position (top of cart)
    let joint_pos = cart_pos + Vec3::new(0.0, config.cart_size.y, 0.0);
    
    // Bottom of pole should be at joint
    let pole_bottom_y = pole_pos.y - pole_half_height;
    
    println!("Joint Y: {}", joint_pos.y);
    println!("Pole bottom Y: {}", pole_bottom_y);
    println!("Difference: {}", (pole_bottom_y - joint_pos.y).abs());
    
    assert!(
        (pole_bottom_y - joint_pos.y).abs() < 0.01,
        "Bottom of pole should be at joint. Joint Y: {}, Pole bottom Y: {}",
        joint_pos.y, pole_bottom_y
    );
}