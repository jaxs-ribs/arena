//! Tests for CartPole physics behavior
//! 
//! These tests isolate specific physics issues like:
//! - Cart sliding without applied force
//! - Pole not falling naturally
//! - Friction behavior

use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3, Vec2, Material};
use approx::assert_relative_eq;

#[test]
fn test_cart_should_not_slide_on_flat_ground_with_friction() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane with high friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    sim.planes[plane_idx].material.restitution = 0.0;
    
    // Add cart with friction
    let cart_pos = Vec3::new(0.0, 0.5, 0.0);
    let cart_idx = sim.add_box(cart_pos, Vec3::new(0.5, 0.25, 0.25), Vec3::ZERO);
    sim.boxes[cart_idx].material.friction = 0.8;
    sim.boxes[cart_idx].mass = 1.0;
    
    // Record initial position
    let initial_x = sim.boxes[cart_idx].pos.x;
    
    // Run simulation for 1 second (60 steps at 60Hz)
    for _ in 0..60 {
        sim.step_cpu();
    }
    
    // Cart should not have moved horizontally
    let final_x = sim.boxes[cart_idx].pos.x;
    assert_relative_eq!(final_x, initial_x, epsilon = 0.01);
    
    // Cart should have settled on the ground
    let final_y = sim.boxes[cart_idx].pos.y;
    assert_relative_eq!(final_y, 0.25, epsilon = 0.01); // Half box height
}

#[test]
fn test_pole_should_fall_naturally_without_control() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    
    // Create a simple cartpole with initial angle
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.5, 0.25, 0.25),
        cart_mass: 1.0,
        pole_length: 2.0,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.2, // ~11.5 degrees
        force_magnitude: 10.0,
        failure_angle: 1.57, // 90 degrees
        position_limit: 5.0,
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Record initial pole angle
    let initial_angle = cartpole.get_pole_angle(&sim);
    assert!(initial_angle.abs() > 0.1); // Should have non-zero initial angle
    
    // Run simulation for 2 seconds without any control
    for _ in 0..120 {
        cartpole.apply_force(&mut sim, 0.0); // No force
        sim.step_cpu();
    }
    
    // Pole should have fallen significantly
    let final_angle = cartpole.get_pole_angle(&sim);
    println!("Pole angle: initial={:.3} rad, final={:.3} rad", initial_angle, final_angle);
    
    // The pole should have increased its angle (fallen further)
    assert!(final_angle.abs() > initial_angle.abs() + 0.5, 
            "Pole didn't fall: initial angle={:.3}, final angle={:.3}", 
            initial_angle, final_angle);
}

#[test]
fn test_revolute_joint_allows_rotation() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Create a box (cart) fixed in place
    let cart_pos = Vec3::new(0.0, 1.0, 0.0);
    let cart_idx = sim.add_box(cart_pos, Vec3::new(0.5, 0.25, 0.25), Vec3::ZERO);
    sim.boxes[cart_idx].mass = 1000.0; // Very heavy to stay in place
    
    // Create a cylinder (pole) that should rotate
    let pole_pos = Vec3::new(0.0, 2.5, 0.0); // Above the cart
    let pole_idx = sim.add_cylinder(pole_pos, 0.1, 1.0, Vec3::ZERO);
    sim.cylinders[pole_idx].mass = 0.1;
    
    // Add revolute joint at top of cart
    let joint_pos = Vec3::new(0.0, 1.25, 0.0); // Top of cart
    let joint_idx = sim.add_revolute_joint(
        0, cart_idx as u32,
        2, pole_idx as u32,
        joint_pos,
        Vec3::new(0.0, 0.0, 1.0) // Z-axis rotation
    );
    
    // Give pole initial horizontal velocity to test rotation
    sim.cylinders[pole_idx].vel = Vec3::new(1.0, 0.0, 0.0);
    
    // Record initial state
    let initial_pole_x = sim.cylinders[pole_idx].pos.x;
    let initial_angular_vel = sim.cylinders[pole_idx].angular_vel.z;
    
    // Run simulation
    for _ in 0..60 {
        sim.step_cpu();
    }
    
    // Check that pole has rotated
    let final_angular_vel = sim.cylinders[pole_idx].angular_vel.z;
    println!("Angular velocity: initial={:.3}, final={:.3}", initial_angular_vel, final_angular_vel);
    
    // Angular velocity should have changed
    assert!(final_angular_vel.abs() > 0.1, 
            "Pole didn't rotate: angular_vel={:.3}", final_angular_vel);
}

#[test]
fn test_cart_with_friction_stops_after_force_removed() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground with friction
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.5;
    
    // Add cart with friction
    let cart_pos = Vec3::new(0.0, 0.5, 0.0);
    let cart_idx = sim.add_box(cart_pos, Vec3::new(0.5, 0.25, 0.25), Vec3::ZERO);
    sim.boxes[cart_idx].material.friction = 0.5;
    sim.boxes[cart_idx].mass = 1.0;
    
    // Apply force for 0.5 seconds
    for _ in 0..30 {
        sim.set_force(cart_idx, [10.0, 0.0]); // 10N to the right
        sim.step_cpu();
    }
    
    // Record velocity after force
    let velocity_with_force = sim.boxes[cart_idx].vel.x;
    assert!(velocity_with_force > 0.5, "Cart should be moving");
    
    // Remove force and let friction stop it
    for _ in 0..60 {
        sim.set_force(cart_idx, [0.0, 0.0]); // No force
        sim.step_cpu();
    }
    
    // Cart should have stopped due to friction
    let final_velocity = sim.boxes[cart_idx].vel.x;
    assert!(final_velocity.abs() < 0.1, 
            "Cart didn't stop: velocity={:.3}", final_velocity);
}

#[test]
fn test_cartpole_pole_falls_with_reduced_joint_stiffness() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground
    let plane_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    sim.planes[plane_idx].material.friction = 0.8;
    
    // Create cartpole
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.5, 0.25, 0.25),
        cart_mass: 1.0,
        pole_length: 2.0,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1, // Small initial angle
        force_magnitude: 10.0,
        failure_angle: 1.57,
        position_limit: 5.0,
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Make cart heavy and add friction to prevent sliding
    sim.boxes[cartpole.cart_idx].mass = 10.0;
    sim.boxes[cartpole.cart_idx].material.friction = 0.8;
    
    // Record initial state
    let initial_angle = cartpole.get_pole_angle(&sim);
    let initial_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Run for 3 seconds
    for i in 0..180 {
        sim.step_cpu();
        
        // Check periodically
        if i % 30 == 0 {
            let angle = cartpole.get_pole_angle(&sim);
            let cart_x = sim.boxes[cartpole.cart_idx].pos.x;
            println!("t={:.1}s: angle={:.3} rad ({:.1}°), cart_x={:.3}", 
                     i as f32 / 60.0, angle, angle.to_degrees(), cart_x);
        }
    }
    
    let final_angle = cartpole.get_pole_angle(&sim);
    let final_cart_x = sim.boxes[cartpole.cart_idx].pos.x;
    
    // Cart should not have moved much
    assert!(final_cart_x.abs() < 0.5, 
            "Cart slid too much: moved {:.3}m", final_cart_x);
    
    // Pole should have fallen
    assert!(final_angle.abs() > 0.5, 
            "Pole didn't fall enough: angle only changed to {:.3} rad ({:.1}°)", 
            final_angle, final_angle.to_degrees());
}