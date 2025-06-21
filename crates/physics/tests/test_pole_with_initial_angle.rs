use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn test_pole_with_actual_initial_angle() {
    let mut sim = PhysicsSim::new();
    
    // Use the actual config from the app
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.05,
        pole_mass: 0.1,
        initial_angle: 0.1, // This is what the app uses!
        force_magnitude: 2.0,
        failure_angle: 1.5,
        position_limit: 3.0,
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    println!("=== INITIAL STATE (angle = 0.1 rad) ===");
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let pole_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    let joint_y = cart_pos.y + config.cart_size.y;
    
    println!("Cart: pos={:?}, top_y={}", cart_pos, joint_y);
    println!("Pole: pos={:?}", pole_pos);
    println!("Pole half_height: {}", sim.cylinders[cartpole.pole_idx].half_height);
    
    let angle_from_quat = 2.0 * pole_orientation[2].atan2(pole_orientation[3]);
    println!("Initial angle from quaternion: {} rad ({} deg)", angle_from_quat, angle_from_quat.to_degrees());
    
    // For tilted pole, where should it be?
    let pole_half_length = config.pole_length / 2.0;
    let expected_x = joint_y + pole_half_length * config.initial_angle.sin();
    let expected_y = joint_y + pole_half_length * config.initial_angle.cos();
    println!("Expected pole center: ({}, {})", expected_x, expected_y);
    println!("Actual pole center: ({}, {})", pole_pos.x, pole_pos.y);
    
    // Check bottom position
    let pole_bottom = pole_pos - Vec3::new(
        pole_half_length * config.initial_angle.sin(),
        pole_half_length * config.initial_angle.cos(),
        0.0
    );
    println!("Calculated pole bottom: {:?}", pole_bottom);
    println!("Should be at joint: (0, {}, 0)", joint_y);
    
    // Run physics and see what happens
    println!("\n=== PHYSICS SIMULATION ===");
    for i in 1..=10 {
        sim.step_cpu();
        
        let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
        let pole_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        let angle = 2.0 * pole_orientation[2].atan2(pole_orientation[3]);
        let angular_vel = sim.cylinders[cartpole.pole_idx].angular_vel.z;
        
        if i <= 3 || i % 3 == 0 {
            println!("\nStep {}: angle={:.3} rad ({:.1}°), angular_vel={:.3}", 
                     i, angle, angle.to_degrees(), angular_vel);
            println!("  Pole pos: {:?}", pole_pos);
            
            // Where is the bottom?
            let bottom_offset = Vec3::new(
                pole_half_length * angle.sin(),
                -pole_half_length * angle.cos(), // Negative because we measure from center DOWN
                0.0
            );
            let bottom_pos = pole_pos + bottom_offset;
            println!("  Bottom pos: {:?} (joint at y={})", bottom_pos, joint_y);
        }
    }
}