use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn debug_pole_hanging_issue() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        initial_angle: 0.0, // Start perfectly vertical
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    println!("=== INITIAL STATE ===");
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let pole_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    let joint_y = cart_pos.y + config.cart_size.y;
    
    println!("Cart: pos={:?}, top_y={}", cart_pos, joint_y);
    println!("Pole: pos={:?}", pole_pos);
    println!("Pole orientation: {:?}", pole_orientation);
    println!("Pole bottom Y: {}", pole_pos.y - sim.cylinders[cartpole.pole_idx].half_height);
    println!("Joint Y: {}", joint_y);
    
    // The problem might be how we calculate the pole position in creation vs constraint
    let angle_from_quat = 2.0 * pole_orientation[2].atan2(pole_orientation[3]);
    println!("Angle from quaternion: {} rad ({} deg)", angle_from_quat, angle_from_quat.to_degrees());
    
    // Check: For angle=0, pole should be straight up
    let expected_pole_center = Vec3::new(
        joint_y + config.pole_length / 2.0 * 0.0, // sin(0)
        joint_y + config.pole_length / 2.0 * 1.0, // cos(0)
        0.0
    );
    println!("Expected pole center for vertical: {:?}", expected_pole_center);
    println!("Actual pole center: {:?}", pole_pos);
    
    // Run a few physics steps
    for i in 1..=5 {
        sim.step_cpu();
        
        let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
        let pole_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        let angle = 2.0 * pole_orientation[2].atan2(pole_orientation[3]);
        
        println!("\n=== STEP {} ===", i);
        println!("Pole pos: {:?}", pole_pos);
        println!("Angle: {} rad ({} deg)", angle, angle.to_degrees());
        println!("Angular vel Z: {}", sim.cylinders[cartpole.pole_idx].angular_vel.z);
        
        // Check where the pole bottom is
        let pole_bottom_y = pole_pos.y - sim.cylinders[cartpole.pole_idx].half_height;
        println!("Pole bottom Y: {}, Joint Y: {}", pole_bottom_y, joint_y);
        println!("Distance from joint: {}", (pole_bottom_y - joint_y).abs());
    }
}