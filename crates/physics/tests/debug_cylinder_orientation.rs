use physics::{PhysicsSim, types::{Vec3, BodyType}};

#[test]
fn test_cylinder_orientation_integration() {
    let mut sim = PhysicsSim::new();
    
    // Create a simple cylinder
    let cylinder_idx = sim.add_cylinder_with_type(
        Vec3::new(0.0, 0.0, 0.0), // position
        0.1,  // radius
        1.0,  // half_height
        Vec3::ZERO, // velocity
        BodyType::Dynamic
    );
    
    // Manually set angular velocity around Z-axis
    sim.cylinders[cylinder_idx].angular_vel.z = 1.0; // 1 rad/s
    
    println!("=== Initial State ===");
    println!("Position: {:?}", sim.cylinders[cylinder_idx].pos);
    println!("Orientation: {:?}", sim.cylinders[cylinder_idx].orientation);
    println!("Angular Velocity: {:?}", sim.cylinders[cylinder_idx].angular_vel);
    
    // Step the simulation multiple times
    for step in 1..=10 {
        sim.step_cpu();
        
        println!("\n=== After Step {} ===", step);
        println!("Position: {:?}", sim.cylinders[cylinder_idx].pos);
        println!("Orientation: {:?}", sim.cylinders[cylinder_idx].orientation);
        println!("Angular Velocity: {:?}", sim.cylinders[cylinder_idx].angular_vel);
        
        // Calculate angle from quaternion
        let q = sim.cylinders[cylinder_idx].orientation;
        let angle_z = 2.0 * (q[3] * q[2] + q[0] * q[1]).atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
        println!("Z-axis rotation from quaternion: {:.3} rad ({:.1}°)", angle_z, angle_z.to_degrees());
    }
    
    // Check that orientation actually changed
    let final_q = sim.cylinders[cylinder_idx].orientation;
    let initial_q = [0.0, 0.0, 0.0, 1.0];
    
    // The quaternion should have changed from initial identity
    assert_ne!(final_q, initial_q, "Cylinder orientation should have changed during integration");
    
    // Check that we have some rotation around Z-axis
    let angle_z = 2.0 * (final_q[3] * final_q[2] + final_q[0] * final_q[1]).atan2(1.0 - 2.0 * (final_q[1] * final_q[1] + final_q[2] * final_q[2]));
    assert!(angle_z.abs() > 0.01, "Should have significant Z-axis rotation, got {:.3} rad", angle_z);
    
    println!("\n✅ Test passed: Cylinder orientation integration is working");
}