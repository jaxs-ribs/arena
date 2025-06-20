//! Tests for revolute joint constraints
//! Following TDD principles - write tests first, then implement

use physics::{
    PhysicsSim,
    types::{Vec3, Vec2},
};

/// Test that a revolute joint keeps two bodies connected at their anchor points
#[test]
fn test_revolute_joint_maintains_connection() {
    println!("\n=== Testing Revolute Joint Connection ===");
    
    let mut sim = PhysicsSim::new();
    
    // Disable gravity for this test
    sim.params.gravity = Vec3::ZERO;
    
    // Create a box (cart) at origin
    let cart_pos = Vec3::new(0.0, 0.0, 0.0);
    let cart_half_extents = Vec3::new(1.0, 0.5, 0.5);
    let cart_idx = sim.add_box(cart_pos, cart_half_extents, Vec3::ZERO);
    println!("Created cart at {:?}", cart_pos);
    
    // Add revolute joint connecting them
    let anchor_on_cart = Vec3::new(0.5, 0.5, 0.0); // Top-right of cart
    let pole_radius = 0.1;
    let pole_half_height = 1.0;
    let anchor_on_pole = Vec3::new(0.0, -pole_half_height, 0.0); // Bottom of pole
    
    // World position of joint (cart position + local anchor)
    let joint_world_pos = cart_pos + anchor_on_cart;
    
    // Position pole so its anchor aligns with the joint
    let pole_pos = joint_world_pos - anchor_on_pole;
    let pole_idx = sim.add_cylinder(pole_pos, pole_radius, pole_half_height, Vec3::ZERO);
    println!("Created pole at {:?}", pole_pos);
    
    let joint_axis = Vec3::new(0.0, 0.0, 1.0); // Rotate around Z axis
    
    let joint_idx = sim.add_revolute_joint(
        0, cart_idx as u32,  // Box type, cart index
        2, pole_idx as u32,  // Cylinder type, pole index
        joint_world_pos,
        joint_axis
    );
    println!("Added revolute joint");
    
    // Run simulation for a few steps
    println!("\nRunning simulation...");
    for step in 0..10 {
        sim.step_cpu();
        
        // Check that anchor points are close together
        let cart_anchor_world = sim.boxes[cart_idx].pos + anchor_on_cart;
        let pole_anchor_world = sim.cylinders[pole_idx].pos + anchor_on_pole;
        let distance = (cart_anchor_world - pole_anchor_world).length();
        
        println!("Step {}: Anchor distance = {:.6}", step, distance);
        
        // They should be very close (within numerical tolerance)
        assert!(distance < 0.01, "Anchor points separated by {} at step {}", distance, step);
    }
    
    println!("✓ Revolute joint maintained connection!");
}

/// Test that revolute joint allows rotation around its axis
#[test]
fn test_revolute_joint_allows_rotation() {
    println!("\n=== Testing Revolute Joint Rotation ===");
    
    let mut sim = PhysicsSim::new();
    
    // Enable gravity - pole should swing down
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Create a static box (cart) - give it huge mass
    let cart_pos = Vec3::new(0.0, 0.0, 0.0);
    let cart_half_extents = Vec3::new(1.0, 0.5, 0.5);
    let cart_idx = sim.add_box(cart_pos, cart_half_extents, Vec3::ZERO);
    sim.boxes[cart_idx].mass = 1000000.0; // Make it effectively static
    println!("Created heavy cart at {:?}", cart_pos);
    
    // Connect with revolute joint at top of cart, bottom of pole
    let anchor_on_cart = Vec3::new(0.0, 0.5, 0.0); // Top center of cart
    let pole_radius = 0.1;
    let pole_half_height = 1.0;
    let anchor_on_pole = Vec3::new(0.0, -pole_half_height, 0.0); // Bottom of pole
    
    // World position of joint
    let joint_world_pos = cart_pos + anchor_on_cart;
    
    // Position pole so its bottom is at the joint, pointing up
    let pole_pos = joint_world_pos - anchor_on_pole;
    let pole_idx = sim.add_cylinder(pole_pos, pole_radius, pole_half_height, Vec3::ZERO);
    sim.cylinders[pole_idx].mass = 0.1; // Light pole
    println!("Created light pole at {:?}", pole_pos);
    
    let joint_axis = Vec3::new(0.0, 0.0, 1.0); // Rotate around Z axis
    
    sim.add_revolute_joint(
        0, cart_idx as u32,
        2, pole_idx as u32,
        joint_world_pos,
        joint_axis
    );
    
    // Record initial pole position
    let initial_pole_top = sim.cylinders[pole_idx].pos + Vec3::new(0.0, pole_half_height, 0.0);
    println!("Initial pole top position: {:?}", initial_pole_top);
    
    // Run simulation - pole should fall/swing
    println!("\nRunning simulation with gravity...");
    for step in 0..50 {
        sim.step_cpu();
        
        if step % 10 == 0 {
            let pole_pos = sim.cylinders[pole_idx].pos;
            let pole_top = pole_pos + Vec3::new(0.0, pole_half_height, 0.0);
            println!("Step {}: Pole position = {:?}, top = {:?}", step, pole_pos, pole_top);
        }
    }
    
    // Check that pole has moved (fallen due to gravity)
    let final_pole_top = sim.cylinders[pole_idx].pos + Vec3::new(0.0, pole_half_height, 0.0);
    let movement = (final_pole_top - initial_pole_top).length();
    
    println!("\nPole top moved by: {:.3} units", movement);
    assert!(movement > 0.5, "Pole should have fallen significantly, but only moved {}", movement);
    
    println!("✓ Revolute joint allowed rotation!");
}

/// Test that force can be applied to cart
#[test]
fn test_cart_force_application() {
    println!("\n=== Testing Force Application to Cart ===");
    
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::ZERO; // No gravity for this test
    
    // Create a box (cart)
    let cart_pos = Vec3::new(0.0, 0.0, 0.0);
    let cart_half_extents = Vec3::new(1.0, 0.5, 0.5);
    let cart_idx = sim.add_box(cart_pos, cart_half_extents, Vec3::ZERO);
    println!("Created cart at {:?}", cart_pos);
    
    // Apply force to the right
    sim.set_force(cart_idx, [10.0, 0.0]); // 10N in +X direction
    println!("Applied 10N force in +X direction");
    
    // Run simulation
    println!("\nRunning simulation...");
    for step in 0..20 {
        sim.step_cpu();
        
        if step % 5 == 0 {
            let pos = sim.boxes[cart_idx].pos;
            let vel = sim.boxes[cart_idx].vel;
            println!("Step {}: Position = ({:.3}, {:.3}, {:.3}), Velocity = ({:.3}, {:.3}, {:.3})", 
                     step, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z);
        }
    }
    
    // Check that cart moved in positive X direction
    let final_pos = sim.boxes[cart_idx].pos;
    println!("\nFinal position: {:?}", final_pos);
    assert!(final_pos.x > 0.1, "Cart should have moved in +X direction, but is at x={}", final_pos.x);
    
    println!("✓ Force application to cart works!");
}

/// Integration test: Simple cartpole that should fall over
#[test]
fn test_cartpole_falls_over() {
    println!("\n=== Testing Cartpole Falling Over ===");
    
    let mut sim = PhysicsSim::new();
    
    // Create cart on ground
    let cart_pos = Vec3::new(0.0, 0.5, 0.0); // Half a meter up (cart height)
    let cart_half_extents = Vec3::new(0.5, 0.25, 0.25);
    let cart_idx = sim.add_box(cart_pos, cart_half_extents, Vec3::ZERO);
    sim.boxes[cart_idx].mass = 1.0;
    println!("Created cart: mass = 1.0 kg");
    
    // Connect with revolute joint
    let anchor_on_cart = Vec3::new(0.0, 0.25, 0.0); // Top of cart
    let pole_angle: f32 = 0.1; // radians from vertical
    let pole_length = 2.0; // 2 meter long pole
    let pole_radius = 0.05;
    let pole_half_height = pole_length / 2.0;
    let anchor_on_pole = Vec3::new(0.0, -pole_half_height, 0.0); // Bottom of pole
    
    // World position of joint
    let joint_world_pos = cart_pos + anchor_on_cart;
    
    // Position pole with slight tilt
    let pole_offset_x = pole_angle.sin() * pole_half_height;
    let pole_offset_y = pole_angle.cos() * pole_half_height;
    let pole_pos = Vec3::new(
        joint_world_pos.x + pole_offset_x,
        joint_world_pos.y + pole_offset_y,
        joint_world_pos.z
    );
    
    let pole_idx = sim.add_cylinder(pole_pos, pole_radius, pole_half_height, Vec3::ZERO);
    sim.cylinders[pole_idx].mass = 0.1;
    println!("Created pole: mass = 0.1 kg, initial angle = {:.2} rad", pole_angle);
    sim.add_revolute_joint(
        0, cart_idx as u32,
        2, pole_idx as u32,
        joint_world_pos,
        Vec3::new(0.0, 0.0, 1.0)
    );
    
    // Add ground plane
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));
    
    // Run simulation and track pole angle
    println!("\nRunning simulation (pole should fall)...");
    for step in 0..100 {
        sim.step_cpu();
        
        if step % 20 == 0 {
            // Calculate pole angle from vertical
            let pole_center = sim.cylinders[pole_idx].pos;
            let cart_center = sim.boxes[cart_idx].pos;
            let joint_pos = cart_center + anchor_on_cart;
            
            let pole_vector = pole_center - joint_pos;
            let angle = pole_vector.x.atan2(pole_vector.y); // Angle from vertical
            
            println!("Step {}: Pole angle = {:.3} rad ({:.1}°), Cart X = {:.3}", 
                     step, angle, angle.to_degrees(), cart_center.x);
        }
    }
    
    // Check final pole angle - should have fallen significantly
    let final_pole_center = sim.cylinders[pole_idx].pos;
    let final_cart_center = sim.boxes[cart_idx].pos;
    let final_joint_pos = final_cart_center + anchor_on_cart;
    let final_pole_vector = final_pole_center - final_joint_pos;
    let final_angle = final_pole_vector.x.atan2(final_pole_vector.y).abs();
    
    println!("\nFinal pole angle: {:.3} rad ({:.1}°)", final_angle, final_angle.to_degrees());
    assert!(final_angle > 0.5, "Pole should have fallen significantly, but angle is only {:.3} rad", final_angle);
    
    println!("✓ Cartpole fell over as expected!");
}