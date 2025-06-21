use glam::{Quat, Vec3};

#[test]
fn test_quaternion_rotate_point_matches_shader() {
    // Test the exact same quaternion rotation formula used in the shader
    
    // Shader formula: p + 2.0 * cross(qv, cross(qv, p) + qw * p)
    fn shader_rotate(p: Vec3, q: [f32; 4]) -> Vec3 {
        let qv = Vec3::new(q[0], q[1], q[2]);
        let qw = q[3];
        p + 2.0 * qv.cross(qv.cross(p) + qw * p)
    }
    
    // Test with a 90 degree rotation around Z axis
    let quat = [0.0, 0.0, 0.707, 0.707]; // 90° around Z
    let point = Vec3::new(0.0, 1.0, 0.0); // Y-axis unit vector
    
    let shader_result = shader_rotate(point, quat);
    println!("Shader rotation of Y-axis by 90° around Z: {:?}", shader_result);
    
    // Using glam for comparison
    let glam_quat = Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]);
    let glam_result = glam_quat * point;
    println!("Glam rotation of Y-axis by 90° around Z: {:?}", glam_result);
    
    // Expected: [-1, 0, 0] (Y rotates to -X)
    assert!((shader_result.x - (-1.0)).abs() < 0.001, "X should be -1, got {}", shader_result.x);
    assert!(shader_result.y.abs() < 0.001, "Y should be 0, got {}", shader_result.y);
    assert!(shader_result.z.abs() < 0.001, "Z should be 0, got {}", shader_result.z);
}

#[test] 
fn test_inverse_rotation_for_local_space() {
    // In the shader, we use quaternion_conjugate to get the inverse rotation
    // Then apply it to transform world space to local space
    
    // If cylinder is rotated 30° around Z, we need to rotate points -30° to get to local space
    let cylinder_rotation = [0.0, 0.0, 0.259, 0.966]; // 30° around Z
    let conjugate = [-cylinder_rotation[0], -cylinder_rotation[1], -cylinder_rotation[2], cylinder_rotation[3]];
    
    // Point that would be at the top of a rotated cylinder
    let world_point = Vec3::new(0.5, 0.866, 0.0); // Top of cylinder rotated 30°
    
    // Shader formula with conjugate
    let qv = Vec3::new(conjugate[0], conjugate[1], conjugate[2]);
    let qw = conjugate[3];
    let local_point = world_point + 2.0 * qv.cross(qv.cross(world_point) + qw * world_point);
    
    println!("World point {:?} transformed to local space: {:?}", world_point, local_point);
    
    // In local space, this should be approximately (0, 1, 0) - top of vertical cylinder
    assert!(local_point.x.abs() < 0.001, "Local X should be ~0, got {}", local_point.x);
    assert!((local_point.y - 1.0).abs() < 0.001, "Local Y should be ~1, got {}", local_point.y);
    assert!(local_point.z.abs() < 0.001, "Local Z should be ~0, got {}", local_point.z);
}

#[test]
fn test_identity_quaternion() {
    // Identity quaternion should not change the point
    let identity = [0.0, 0.0, 0.0, 1.0];
    let point = Vec3::new(1.0, 2.0, 3.0);
    
    let qv = Vec3::new(identity[0], identity[1], identity[2]);
    let qw = identity[3];
    let rotated = point + 2.0 * qv.cross(qv.cross(point) + qw * point);
    
    println!("Identity rotation of {:?}: {:?}", point, rotated);
    
    assert_eq!(rotated, point, "Identity quaternion should not change the point");
}