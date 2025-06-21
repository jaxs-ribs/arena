use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn test_pole_stays_attached_during_motion() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 1.5,
        pole_radius: 0.05,
        initial_angle: 0.1,
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Apply force to move cart
    cartpole.apply_force(&mut sim, 1.0); // Move right
    
    // Run simulation for several steps
    for i in 0..20 {
        sim.step_cpu();
        
        // Get cart and pole positions
        let cart_pos = sim.boxes[cartpole.cart_idx].pos;
        let joint_pos = cart_pos + Vec3::new(0.0, sim.boxes[cartpole.cart_idx].half_extents.y, 0.0);
        
        let pole_center = sim.cylinders[cartpole.pole_idx].pos;
        let pole_half_height = sim.cylinders[cartpole.pole_idx].half_height;
        
        // Calculate pole bottom from center and angle
        let angle = cartpole.get_pole_angle(&sim);
        let pole_bottom = pole_center - Vec3::new(
            pole_half_height * angle.sin(),
            pole_half_height * angle.cos(),
            0.0
        );
        
        // Check that pole bottom is at joint
        let distance_to_joint = (pole_bottom - joint_pos).length();
        
        println!("Step {}: Cart x={:.3}, Joint=({:.3}, {:.3}), Pole bottom=({:.3}, {:.3}), Distance={:.6}",
                 i, cart_pos.x, joint_pos.x, joint_pos.y, pole_bottom.x, pole_bottom.y, distance_to_joint);
        
        // Assert pole stays attached (within small tolerance for numerical errors)
        assert!(distance_to_joint < 0.001, 
                "Pole detached from cart at step {}! Distance: {}", i, distance_to_joint);
    }
}

#[test]
fn test_pole_attachment_with_alternating_forces() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        pole_radius: 0.05,
        initial_angle: 0.05,
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Apply alternating forces
    for i in 0..30 {
        // Alternate between left and right
        let force = if i % 10 < 5 { 1.0 } else { -1.0 };
        cartpole.apply_force(&mut sim, force);
        
        sim.step_cpu();
        
        // Check attachment
        let cart_pos = sim.boxes[cartpole.cart_idx].pos;
        let joint_pos = cart_pos + Vec3::new(0.0, sim.boxes[cartpole.cart_idx].half_extents.y, 0.0);
        
        let pole_center = sim.cylinders[cartpole.pole_idx].pos;
        let pole_half_height = sim.cylinders[cartpole.pole_idx].half_height;
        let angle = cartpole.get_pole_angle(&sim);
        
        let pole_bottom = pole_center - Vec3::new(
            pole_half_height * angle.sin(),
            pole_half_height * angle.cos(),
            0.0
        );
        
        let distance = (pole_bottom - joint_pos).length();
        assert!(distance < 0.001, "Pole detached during alternating forces at step {}", i);
    }
}

#[test]
fn test_pole_physics_response_to_cart_motion() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 1.5,
        pole_radius: 0.05,
        initial_angle: 0.01, // Nearly vertical
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Get initial angle
    let initial_angle = cartpole.get_pole_angle(&sim);
    
    // Apply rightward force to cart
    cartpole.apply_force(&mut sim, 1.0);
    
    // Run a few steps
    for _ in 0..5 {
        sim.step_cpu();
    }
    
    let angle_after_right = cartpole.get_pole_angle(&sim);
    
    // When cart moves right, pole should tilt left (negative angle change)
    assert!(angle_after_right < initial_angle, 
            "Pole should tilt left when cart moves right. Initial: {:.3}, After: {:.3}", 
            initial_angle, angle_after_right);
    
    // Now apply leftward force
    cartpole.apply_force(&mut sim, -1.0);
    
    // Run more steps
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    let angle_after_left = cartpole.get_pole_angle(&sim);
    
    // Pole should now be tilting the other way
    assert!(angle_after_left > angle_after_right,
            "Pole should respond to cart direction change. Was: {:.3}, Now: {:.3}",
            angle_after_right, angle_after_left);
}

#[test]
fn test_cart_velocity_affects_pole() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 1.5,
        pole_radius: 0.05,
        initial_angle: 0.0, // Perfectly vertical
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Move cart at constant velocity
    cartpole.apply_force(&mut sim, 1.0);
    
    // Let it accelerate
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    // Check that cart is moving
    let cart_vel = sim.boxes[cartpole.cart_idx].vel;
    assert!(cart_vel.x > 0.1, "Cart should be moving right");
    
    // Check that pole has tilted due to cart motion
    let angle = cartpole.get_pole_angle(&sim);
    assert!(angle.abs() > 0.01, "Pole should have tilted due to cart motion");
}

#[test]
fn test_pole_falls_naturally_without_control() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 1.5,
        pole_radius: 0.05,
        initial_angle: 0.1, // Small initial tilt
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    let initial_angle = cartpole.get_pole_angle(&sim);
    
    // Run simulation without any control
    for _ in 0..20 {
        sim.step_cpu();
    }
    
    let final_angle = cartpole.get_pole_angle(&sim);
    
    // Pole should fall in the direction of initial tilt
    assert!(final_angle > initial_angle,
            "Pole should fall over time. Initial: {:.3}, Final: {:.3}",
            initial_angle, final_angle);
    assert!(final_angle > 0.5, "Pole should have fallen significantly");
}