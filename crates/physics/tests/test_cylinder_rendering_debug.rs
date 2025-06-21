use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn debug_cylinder_rendering_data() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 1.5,
        pole_radius: 0.05,
        initial_angle: 0.1, // 5.7 degrees
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    println!("\n=== INITIAL STATE ===");
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let cart_half_extents = sim.boxes[cartpole.cart_idx].half_extents;
    let joint_pos = cart_pos + Vec3::new(0.0, cart_half_extents.y, 0.0);
    
    let pole = &sim.cylinders[cartpole.pole_idx];
    println!("Cart position: {:?}", cart_pos);
    println!("Cart half_extents: {:?}", cart_half_extents);
    println!("Joint position: {:?}", joint_pos);
    println!("\nPole data:");
    println!("  pos (center): {:?}", pole.pos);
    println!("  radius: {}", pole.radius);
    println!("  half_height: {}", pole.half_height);
    println!("  orientation: {:?}", pole.orientation);
    
    // Calculate where the bottom and top should be
    let angle = cartpole.get_pole_angle(&sim);
    let direction = Vec3::new(angle.sin(), angle.cos(), 0.0);
    let pole_bottom = pole.pos - direction * pole.half_height;
    let pole_top = pole.pos + direction * pole.half_height;
    
    println!("\nCalculated positions:");
    println!("  Angle: {} rad ({} deg)", angle, angle.to_degrees());
    println!("  Direction vector: {:?}", direction);
    println!("  Pole bottom: {:?}", pole_bottom);
    println!("  Pole top: {:?}", pole_top);
    println!("  Distance from bottom to joint: {}", (pole_bottom - joint_pos).length());
    
    // What would be sent to GPU
    println!("\n=== GPU DATA (what renderer receives) ===");
    println!("Position: [{:.6}, {:.6}, {:.6}]", pole.pos.x, pole.pos.y, pole.pos.z);
    println!("Height: {:.6}", pole.half_height * 2.0);
    println!("Radius: {:.6}", pole.radius);
    println!("Orientation: [{:.6}, {:.6}, {:.6}, {:.6}]", 
             pole.orientation[0], pole.orientation[1], pole.orientation[2], pole.orientation[3]);
    
    // Now run a few physics steps
    println!("\n=== AFTER 10 PHYSICS STEPS ===");
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    let pole = &sim.cylinders[cartpole.pole_idx];
    let angle = cartpole.get_pole_angle(&sim);
    println!("New angle: {} rad ({} deg)", angle, angle.to_degrees());
    println!("New position: {:?}", pole.pos);
    println!("New orientation: {:?}", pole.orientation);
    
    // Verify quaternion represents the same angle
    let qz = pole.orientation[2];
    let qw = pole.orientation[3];
    let angle_from_quat = 2.0 * qz.atan2(qw);
    println!("\nQuaternion verification:");
    println!("  Angle from physics: {}", angle);
    println!("  Angle from quaternion: {}", angle_from_quat);
    println!("  Difference: {}", (angle - angle_from_quat).abs());
}