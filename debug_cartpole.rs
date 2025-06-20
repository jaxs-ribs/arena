use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3};

fn main() {
    println!("=== Debugging CartPole Angle Calculation ===");
    
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.05,
        pole_mass: 0.1,
        initial_angle: 0.05, // 2.9 degrees
        force_magnitude: 10.0,
        failure_angle: 0.5, // 28.6 degrees
        position_limit: 3.0,
    };
    
    println!("Config:");
    println!("  Initial angle: {} rad ({:.1}°)", config.initial_angle, config.initial_angle.to_degrees());
    println!("  Failure angle: {} rad ({:.1}°)", config.failure_angle, config.failure_angle.to_degrees());
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Check initial state
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let joint_pos = cart_pos + Vec3::new(0.0, config.cart_size.y, 0.0);
    let pole_vector = pole_pos - joint_pos;
    
    println!("\nInitial positions:");
    println!("  Cart pos: ({:.3}, {:.3}, {:.3})", cart_pos.x, cart_pos.y, cart_pos.z);
    println!("  Pole pos: ({:.3}, {:.3}, {:.3})", pole_pos.x, pole_pos.y, pole_pos.z);
    println!("  Joint pos: ({:.3}, {:.3}, {:.3})", joint_pos.x, joint_pos.y, joint_pos.z);
    println!("  Pole vector: ({:.3}, {:.3}, {:.3})", pole_vector.x, pole_vector.y, pole_vector.z);
    
    let angle = pole_vector.x.atan2(pole_vector.y);
    println!("  Calculated angle: {:.4} rad ({:.1}°)", angle, angle.to_degrees());
    
    let state = cartpole.get_state(&sim);
    println!("\nState vector: [{:.3}, {:.3}, {:.3}, {:.3}]", state[0], state[1], state[2], state[3]);
    
    // Run one step
    println!("\nRunning one physics step...");
    sim.step_cpu();
    
    let cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let pole_pos = sim.cylinders[cartpole.pole_idx].pos;
    let joint_pos = cart_pos + Vec3::new(0.0, config.cart_size.y, 0.0);
    let pole_vector = pole_pos - joint_pos;
    let angle = pole_vector.x.atan2(pole_vector.y);
    
    println!("After one step:");
    println!("  Pole vector: ({:.3}, {:.3}, {:.3})", pole_vector.x, pole_vector.y, pole_vector.z);
    println!("  Angle: {:.4} rad ({:.1}°)", angle, angle.to_degrees());
    
    // Check expected pole position for initial angle
    let expected_x = config.initial_angle.sin() * config.pole_length / 2.0;
    let expected_y = config.initial_angle.cos() * config.pole_length / 2.0;
    println!("\nExpected pole offset from joint:");
    println!("  X: {:.3}, Y: {:.3}", expected_x, expected_y);
    println!("  Expected angle from offsets: {:.4} rad", expected_x.atan2(expected_y));
}