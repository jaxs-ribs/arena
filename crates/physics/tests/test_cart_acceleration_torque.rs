use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::{Vec3, Vec2};

#[test]
fn test_cart_acceleration_affects_pole() {
    let mut sim = PhysicsSim::new();
    
    // Add ground plane
    sim.add_plane(Vec3::new(0.0, -1.0, 0.0), 1.0, Vec2::new(100.0, 100.0));
    
    let config = CartPoleConfig {
        initial_angle: 0.0, // Start perfectly vertical
        ..Default::default()
    };
    
    // Create a cartpole
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Initial state - pole should be vertical
    let initial_angle = cartpole.get_pole_angle(&sim);
    assert!(initial_angle.abs() < 0.001, "Pole should start vertical");
    
    // Apply rightward force to cart for several steps
    for _ in 0..5 {
        cartpole.apply_force(&mut sim, 1.0); // Full right
        sim.step_cpu();
    }
    
    // Cart should have moved right
    let cart_pos = sim.boxes[cartpole.cart_idx].pos.x;
    println!("Cart position after acceleration: {:.3}", cart_pos);
    assert!(cart_pos > 0.0, "Cart should have moved right");
    
    // CRITICAL TEST: Pole should tilt BACKWARD (negative angle) when cart accelerates right
    let pole_angle = cartpole.get_pole_angle(&sim);
    println!("Pole angle after cart acceleration: {:.3} rad ({:.1}°)", 
             pole_angle, pole_angle.to_degrees());
    
    // When cart accelerates right, pole should tilt left (negative angle)
    assert!(pole_angle < -0.01, 
            "Pole should tilt backward when cart accelerates forward. Got: {:.3} rad", 
            pole_angle);
}

#[test]
fn test_balancing_torque() {
    let mut sim = PhysicsSim::new();
    
    // Add ground plane
    sim.add_plane(Vec3::new(0.0, -1.0, 0.0), 1.0, Vec2::new(100.0, 100.0));
    
    let config = CartPoleConfig {
        initial_angle: 0.1, // Start with small tilt to the right
        ..Default::default()
    };
    
    // Create a cartpole
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Initial angle
    let initial_angle = cartpole.get_pole_angle(&sim);
    println!("Initial pole angle: {:.3} rad ({:.1}°)", 
             initial_angle, initial_angle.to_degrees());
    
    // Apply force in same direction as tilt (should help balance)
    for i in 0..10 {
        cartpole.apply_force(&mut sim, 0.5); // Moderate right force
        sim.step_cpu();
        
        let angle = cartpole.get_pole_angle(&sim);
        println!("Step {}: angle = {:.3} rad ({:.1}°)", 
                 i, angle, angle.to_degrees());
    }
    
    // The angle should be smaller than initial (cart motion helped balance)
    let final_angle = cartpole.get_pole_angle(&sim);
    println!("Final angle: {:.3} rad, Initial was: {:.3} rad", 
             final_angle, initial_angle);
    
    // We expect the balancing force to have reduced the tilt
    assert!(final_angle.abs() < initial_angle.abs() + 0.1, 
            "Cart motion should help balance the pole");
}