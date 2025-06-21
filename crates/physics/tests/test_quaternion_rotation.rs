use physics::types::Vec3;

fn quaternion_rotate_point(p: Vec3, q: [f32; 4]) -> Vec3 {
    // Mimicking the WGSL shader function
    let qv = Vec3::new(q[0], q[1], q[2]);
    let qw = q[3];
    p + qv.cross(qv.cross(p) + p * qw) * 2.0
}

fn quaternion_conjugate(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

#[test]
fn test_shader_quaternion_math() {
    // Test with the actual orientation from the cartpole
    let orientation = [0.0f32, 0.0f32, 0.049979f32, 0.998750f32]; // ~5.7 degrees around Z
    let angle = 2.0f32 * orientation[2].atan2(orientation[3]);
    println!("Testing with angle: {} rad ({} deg)", angle, angle.to_degrees());
    
    // Test point at the bottom of a cylinder (center at 1.146253, half_height = 0.75)
    let center = Vec3::new(0.074875, 1.146253, 0.0);
    let half_height = 0.75;
    
    // In world space, the bottom should be at (0, 0.4, 0) based on our test
    let expected_bottom = Vec3::new(0.0, 0.4, 0.0);
    
    // Calculate where the shader thinks the bottom is
    // In local space, bottom is at (0, -half_height, 0)
    let local_bottom = Vec3::new(0.0, -half_height, 0.0);
    
    // Transform from local to world using the quaternion
    let rotated_bottom = quaternion_rotate_point(local_bottom, orientation);
    let world_bottom = center + rotated_bottom;
    
    println!("\nLocal bottom: {:?}", local_bottom);
    println!("After rotation: {:?}", rotated_bottom);
    println!("World bottom: {:?}", world_bottom);
    println!("Expected bottom: {:?}", expected_bottom);
    println!("Error: {:?}", world_bottom - expected_bottom);
    
    // Now test the inverse: given a world point, what's its local position?
    let test_point = expected_bottom;
    let offset = test_point - center;
    let local_point = quaternion_rotate_point(offset, quaternion_conjugate(orientation));
    
    println!("\nInverse test:");
    println!("World point: {:?}", test_point);
    println!("Offset from center: {:?}", offset);
    println!("Local point: {:?}", local_point);
    println!("Expected local: {:?}", Vec3::new(0.0, -half_height, 0.0));
    
    // Test SDF calculation at the bottom point
    let d_x = local_point.x.abs();
    let d_z = local_point.z.abs();
    let d_radial = (d_x * d_x + d_z * d_z).sqrt();
    let d_y = local_point.y.abs();
    
    println!("\nSDF components:");
    println!("Radial distance: {}", d_radial);
    println!("Y distance: {}", d_y);
    println!("Y distance - half_height: {}", d_y - half_height);
}