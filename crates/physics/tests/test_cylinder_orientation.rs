use glam::Quat;
use physics::{PhysicsSim, Vec3};
use physics::types::BodyType;

#[test]
fn test_cylinder_orientation_storage() {
    println!("\n=== Testing Cylinder Orientation Storage ===\n");
    
    // Create a physics simulation
    let mut sim = PhysicsSim::new();
    
    // Create a 45-degree rotation around Y axis
    let test_orientation = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    println!("1. Created test quaternion (45° Y rotation):");
    println!("   [{:.3}, {:.3}, {:.3}, {:.3}]", 
             test_orientation.x, test_orientation.y, test_orientation.z, test_orientation.w);
    
    // Create a cylinder
    let cylinder_idx = sim.add_cylinder_with_type(
        Vec3::new(0.0, 0.0, 0.0),
        0.5,  // radius
        1.0,  // half_height
        Vec3::new(0.0, 0.0, 0.0),
        BodyType::Dynamic
    );
    
    // Set the orientation
    sim.cylinders[cylinder_idx].orientation = [
        test_orientation.x,
        test_orientation.y,
        test_orientation.z,
        test_orientation.w
    ];
    
    println!("\n2. Stored in Cylinder struct:");
    let cylinder = &sim.cylinders[cylinder_idx];
    println!("   [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder.orientation[0], cylinder.orientation[1], 
             cylinder.orientation[2], cylinder.orientation[3]);
    
    // Verify the orientation is preserved
    assert_eq!(cylinder.orientation[0], test_orientation.x);
    assert_eq!(cylinder.orientation[1], test_orientation.y);
    assert_eq!(cylinder.orientation[2], test_orientation.z);
    assert_eq!(cylinder.orientation[3], test_orientation.w);
    println!("\n✅ Orientation correctly stored in Cylinder struct!");
    
    // Test the expected rotation matrix
    let rotation_matrix = glam::Mat3::from_quat(test_orientation);
    println!("\n3. Expected rotation matrix:");
    println!("   [{:.3}, {:.3}, {:.3}]", 
             rotation_matrix.x_axis.x, rotation_matrix.x_axis.y, rotation_matrix.x_axis.z);
    println!("   [{:.3}, {:.3}, {:.3}]", 
             rotation_matrix.y_axis.x, rotation_matrix.y_axis.y, rotation_matrix.y_axis.z);
    println!("   [{:.3}, {:.3}, {:.3}]", 
             rotation_matrix.z_axis.x, rotation_matrix.z_axis.y, rotation_matrix.z_axis.z);
    
    // For a 45-degree Y rotation, we expect:
    // cos(45°) ≈ 0.707, sin(45°) ≈ 0.707
    let expected = 0.707;
    let tolerance = 0.001;
    
    assert!((rotation_matrix.x_axis.x - expected).abs() < tolerance);
    assert!((rotation_matrix.x_axis.z - expected).abs() < tolerance);
    assert!((rotation_matrix.z_axis.x - (-expected)).abs() < tolerance);
    assert!((rotation_matrix.z_axis.z - expected).abs() < tolerance);
    
    println!("\n✅ Rotation matrix is correct for 45° Y rotation!");
}