use glam::Quat;
use physics::{Cylinder, Material, Vec3};

#[test]
fn test_cylinder_orientation_storage() {
    println!("\n=== Testing Cylinder Orientation Storage ===\n");
    
    // Create a 45-degree rotation around Y axis
    let test_orientation = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    println!("1. Created test quaternion (45° Y rotation):");
    println!("   [{:.3}, {:.3}, {:.3}, {:.3}]", 
             test_orientation.x, test_orientation.y, test_orientation.z, test_orientation.w);
    
    // Create a cylinder with this orientation
    let cylinder = Cylinder {
        center: Vec3::new(0.0, 0.0, 0.0),
        orientation: test_orientation,
        radius: 0.5,
        half_height: 1.0,
        velocity: Vec3::new(0.0, 0.0, 0.0),
        angular_velocity: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        moment_of_inertia: Vec3::new(1.0, 1.0, 1.0),
        material: Material::new(0.2, 0.3),
        color: [1.0, 0.0, 0.0],
    };
    
    println!("\n2. Stored in Cylinder struct:");
    println!("   [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder.orientation.x, cylinder.orientation.y, 
             cylinder.orientation.z, cylinder.orientation.w);
    
    // Verify the orientation is preserved
    assert_eq!(cylinder.orientation, test_orientation);
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