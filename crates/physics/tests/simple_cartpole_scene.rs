//! Simple CartPole scene test to verify basic behavior

use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3, Vec2};

#[test]
fn test_simple_cartpole_scene() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane with friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    // Create a single cartpole
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1, // ~5.7 degrees
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Print initial state
    println!("\nInitial state:");
    println!("Cart position: {:?}", sim.boxes[cartpole.cart_idx].pos);
    println!("Cart velocity: {:?}", sim.boxes[cartpole.cart_idx].vel);
    println!("Cart friction: {}", sim.boxes[cartpole.cart_idx].material.friction);
    println!("Pole position: {:?}", sim.cylinders[cartpole.pole_idx].pos);
    println!("Pole angle: {:.3} rad ({:.1}°)", 
             cartpole.get_pole_angle(&sim), 
             cartpole.get_pole_angle(&sim).to_degrees());
    
    // Run for 1 second without any control
    println!("\nRunning simulation...");
    for i in 0..60 {
        sim.step_cpu();
        
        if i % 10 == 9 {
            let t = (i + 1) as f32 / 60.0;
            let cart_pos = sim.boxes[cartpole.cart_idx].pos;
            let cart_vel = sim.boxes[cartpole.cart_idx].vel;
            let pole_angle = cartpole.get_pole_angle(&sim);
            let pole_angular_vel = sim.cylinders[cartpole.pole_idx].angular_vel.z;
            
            println!("t={:.2}s: cart_x={:.3}, cart_vx={:.3}, pole_angle={:.3} rad ({:.1}°), pole_ω={:.3}", 
                     t, cart_pos.x, cart_vel.x, pole_angle, pole_angle.to_degrees(), pole_angular_vel);
        }
    }
    
    // Final state
    let final_cart_pos = sim.boxes[cartpole.cart_idx].pos;
    let final_pole_angle = cartpole.get_pole_angle(&sim);
    
    println!("\nFinal state:");
    println!("Cart moved: {:.3}m", final_cart_pos.x);
    println!("Pole angle: {:.3} rad ({:.1}°)", final_pole_angle, final_pole_angle.to_degrees());
    
    // The cart should not have moved much (friction should keep it in place)
    assert!(final_cart_pos.x.abs() < 0.1, "Cart slid too much: {:.3}m", final_cart_pos.x);
    
    // The pole should have fallen (at least 15° or 0.26 rad in 1 second)
    assert!(final_pole_angle.abs() > 0.25, "Pole didn't fall enough: {:.3} rad ({:.1}°)", 
            final_pole_angle, final_pole_angle.to_degrees());
}