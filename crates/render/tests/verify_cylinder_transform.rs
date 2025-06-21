use glam::Vec3;

#[test]
fn verify_cartpole_cylinder_transform() {
    println!("\n=== Verifying CartPole Cylinder Transform ===\n");
    
    // Scenario: CartPole with pole tilted 30° from vertical
    let pole_angle = 30.0_f32.to_radians(); // In X-Y plane
    
    println!("CartPole scenario:");
    println!("- Pole angle: {:.1}° from vertical", pole_angle.to_degrees());
    println!("- Pole swings in X-Y plane (around Z-axis)");
    
    // Physics generates this quaternion for Z-axis rotation
    let half_angle = pole_angle * 0.5;
    let physics_quat = [0.0, 0.0, half_angle.sin(), half_angle.cos()];
    println!("\nPhysics quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             physics_quat[0], physics_quat[1], physics_quat[2], physics_quat[3]);
    
    // In world space, the pole should be:
    // - Base at (0, 0, 0)
    // - Top at (sin(30°), cos(30°), 0) = (0.5, 0.866, 0)
    let expected_top = Vec3::new(pole_angle.sin(), pole_angle.cos(), 0.0);
    println!("\nExpected pole top position: ({:.3}, {:.3}, {:.3})", 
             expected_top.x, expected_top.y, expected_top.z);
    
    // Test shader transform
    // In local space, cylinder is vertical: top at (0, 1, 0)
    let local_top = Vec3::new(0.0, 1.0, 0.0);
    
    // Apply rotation to get world space
    let rotated_top = quaternion_rotate_point(local_top, physics_quat);
    println!("\nShader transform result:");
    println!("- Local top (0, 1, 0) -> World top ({:.3}, {:.3}, {:.3})", 
             rotated_top.x, rotated_top.y, rotated_top.z);
    
    // Check if it matches expected
    let error = (rotated_top - expected_top).length();
    if error < 0.001 {
        println!("✅ Transform is correct!");
    } else {
        println!("❌ Transform error: {:.3}", error);
    }
    
    // Now test the SDF evaluation
    println!("\n--- Testing SDF Evaluation ---");
    
    // Test point: slightly to the right of the tilted pole
    let test_point = Vec3::new(0.3, 0.5, 0.0);
    println!("\nTest point in world space: ({:.3}, {:.3}, {:.3})", 
             test_point.x, test_point.y, test_point.z);
    
    // Transform to local space (inverse rotation)
    let local_test = quaternion_rotate_point_inverse(test_point, physics_quat);
    println!("Test point in cylinder local space: ({:.3}, {:.3}, {:.3})", 
             local_test.x, local_test.y, local_test.z);
    
    // In local space, cylinder has radius 0.05, height 1.5
    // It's centered at origin, extends from y=-0.75 to y=0.75
    let cylinder_radius = 0.05;
    let cylinder_half_height = 0.75;
    
    // Calculate SDF in local space
    let radial_dist = (local_test.x * local_test.x + local_test.z * local_test.z).sqrt();
    let height_dist = local_test.y.abs();
    
    println!("\nLocal space distances:");
    println!("- Radial distance: {:.3} (radius: {:.3})", radial_dist, cylinder_radius);
    println!("- Height distance: {:.3} (half-height: {:.3})", height_dist, cylinder_half_height);
    
    // The SDF should be negative inside, positive outside
    let d_radial = radial_dist - cylinder_radius;
    let d_height = height_dist - cylinder_half_height;
    let sdf = d_radial.max(d_height);
    
    println!("\nSDF value: {:.3}", sdf);
    if sdf < 0.0 {
        println!("Point is INSIDE the cylinder");
    } else {
        println!("Point is OUTSIDE the cylinder");
    }
}

// Quaternion rotation (matches shader)
fn quaternion_rotate_point(p: Vec3, q: [f32; 4]) -> Vec3 {
    let qv = Vec3::new(q[0], q[1], q[2]);
    let qw = q[3];
    p + 2.0 * qv.cross(qv.cross(p) + qw * p)
}

// Inverse rotation using conjugate
fn quaternion_rotate_point_inverse(p: Vec3, q: [f32; 4]) -> Vec3 {
    let conjugate = [-q[0], -q[1], -q[2], q[3]];
    quaternion_rotate_point(p, conjugate)
}