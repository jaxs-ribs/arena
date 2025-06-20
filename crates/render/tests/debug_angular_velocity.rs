use physics::{PhysicsSim, Vec3, CartPole, CartPoleConfig};

#[test]
fn debug_angular_velocity() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0,
        ..Default::default()
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::new(0.0, 0.0, 0.0), config);
    let pole_idx = cartpole.pole_idx;
    
    eprintln!("\nBefore physics step:");
    eprintln!("  Position: {:?}", sim.cylinders[pole_idx].pos);
    eprintln!("  Velocity: {:?}", sim.cylinders[pole_idx].vel);
    eprintln!("  Angular velocity: {:?}", sim.cylinders[pole_idx].angular_vel);
    eprintln!("  Orientation: {:?}", sim.cylinders[pole_idx].orientation);
    eprintln!("  Body type: {:?}", sim.cylinders[pole_idx].body_type);
    
    // Apply a small force to the cart to induce motion
    cartpole.apply_force(&mut sim, 0.1);
    
    // Run multiple physics steps
    for i in 0..5 {
        sim.step_cpu();
        eprintln!("\nAfter step {}:", i + 1);
        eprintln!("  Position: {:?}", sim.cylinders[pole_idx].pos);
        eprintln!("  Angular velocity: {:?}", sim.cylinders[pole_idx].angular_vel);
        eprintln!("  Orientation: {:?}", sim.cylinders[pole_idx].orientation);
        
        // Also check the cart
        eprintln!("  Cart position: {:?}", sim.boxes[cartpole.cart_idx].pos);
    }
    
    // After 5 steps, the pole should have moved/rotated
    let final_orientation = sim.cylinders[pole_idx].orientation;
    eprintln!("\nFinal orientation: {:?}", final_orientation);
    
    // Check if angular velocity ever became non-zero
    let ang_vel = sim.cylinders[pole_idx].angular_vel;
    eprintln!("Final angular velocity: {:?}", ang_vel);
}