use glam::{Vec3, Quat, Mat4};

#[test]
fn test_transform_matrix_orientation() {
    println!("\n=== Testing Transform Matrix Orientation ===");
    
    // Test 1: 45-degree rotation around Z axis
    let rotation = Quat::from_rotation_z(std::f32::consts::PI / 4.0);
    let position = Vec3::new(0.0, 1.5, 0.0);
    let scale = Vec3::ONE;
    
    println!("Test 1: 45° rotation around Z");
    println!("  Rotation quaternion: {:?}", rotation);
    println!("  Expected: ~45 degrees around Z axis");
    
    let matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);
    println!("  Transform matrix:");
    print_matrix(&matrix);
    
    // Extract back the rotation
    let (extracted_scale, extracted_rotation, extracted_position) = matrix.to_scale_rotation_translation();
    println!("  Extracted rotation: {:?}", extracted_rotation);
    println!("  Extracted position: {:?}", extracted_position);
    println!("  Extracted scale: {:?}", extracted_scale);
    
    // Verify rotation is preserved
    assert_quaternions_equal(rotation, extracted_rotation, "Rotation should be preserved in matrix");
    assert!((position - extracted_position).length() < 0.001, "Position should be preserved");
    assert!((scale - extracted_scale).length() < 0.001, "Scale should be preserved");
    
    // Test 2: Multiple different rotations
    println!("\nTest 2: Various rotations");
    let test_rotations = vec![
        ("30° X", Quat::from_rotation_x(std::f32::consts::PI / 6.0)),
        ("60° Y", Quat::from_rotation_y(std::f32::consts::PI / 3.0)),
        ("90° Z", Quat::from_rotation_z(std::f32::consts::PI / 2.0)),
        ("Complex", Quat::from_euler(glam::EulerRot::XYZ, 0.5, 0.7, 0.3)),
    ];
    
    for (desc, rot) in test_rotations {
        println!("\n  Testing {}: {:?}", desc, rot);
        let mat = Mat4::from_scale_rotation_translation(Vec3::ONE, rot, Vec3::ZERO);
        let (_, extracted, _) = mat.to_scale_rotation_translation();
        assert_quaternions_equal(rot, extracted, &format!("{} rotation should be preserved", desc));
        println!("    ✓ Rotation preserved correctly");
    }
    
    // Test 3: Verify non-identity rotations are detectable
    println!("\nTest 3: Identity vs non-identity detection");
    let identity = Quat::IDENTITY;
    let small_rotation = Quat::from_rotation_y(0.1); // Small 0.1 radian rotation
    
    println!("  Identity: {:?} (w={:.4})", identity, identity.w);
    println!("  Small rotation: {:?} (w={:.4})", small_rotation, small_rotation.w);
    
    assert!(identity.w > 0.999, "Identity quaternion should have w very close to 1.0");
    assert!(small_rotation.w < 0.999, "Non-identity quaternion should have w < 0.999");
    
    // Test 4: Column-major vs row-major verification
    println!("\nTest 4: Matrix layout verification");
    let rot = Quat::from_rotation_z(std::f32::consts::PI / 2.0); // 90° rotation
    let mat = Mat4::from_rotation_translation(rot, Vec3::new(1.0, 2.0, 3.0));
    
    let cols = mat.to_cols_array_2d();
    println!("  Matrix columns:");
    for i in 0..4 {
        println!("    Col {}: [{:.3}, {:.3}, {:.3}, {:.3}]", i, cols[i][0], cols[i][1], cols[i][2], cols[i][3]);
    }
    
    // For a 90° Z rotation, we expect:
    // X axis (1,0,0) -> (0,1,0)
    // Y axis (0,1,0) -> (-1,0,0)
    println!("  Expected for 90° Z rotation:");
    println!("    X axis rotated: ~(0, 1, 0)");
    println!("    Y axis rotated: ~(-1, 0, 0)");
    println!("  Actual:");
    println!("    Column 0 (rotated X): ({:.3}, {:.3}, {:.3})", cols[0][0], cols[0][1], cols[0][2]);
    println!("    Column 1 (rotated Y): ({:.3}, {:.3}, {:.3})", cols[1][0], cols[1][1], cols[1][2]);
    
    // Verify the rotation is correct
    assert!((cols[0][0]).abs() < 0.001, "X component of rotated X axis should be ~0");
    assert!((cols[0][1] - 1.0).abs() < 0.001, "Y component of rotated X axis should be ~1");
    assert!((cols[1][0] + 1.0).abs() < 0.001, "X component of rotated Y axis should be ~-1");
    assert!((cols[1][1]).abs() < 0.001, "Y component of rotated Y axis should be ~0");
    
    println!("\n=== All transform tests passed! ===");
}

fn print_matrix(mat: &Mat4) {
    let cols = mat.to_cols_array_2d();
    for row in 0..4 {
        println!("    [{:7.3}, {:7.3}, {:7.3}, {:7.3}]",
            cols[0][row], cols[1][row], cols[2][row], cols[3][row]
        );
    }
}

fn assert_quaternions_equal(q1: Quat, q2: Quat, message: &str) {
    // Quaternions q and -q represent the same rotation
    let dot = q1.dot(q2);
    let same = dot.abs() > 0.999; // Very close to 1 or -1
    
    assert!(same, "{}: q1={:?}, q2={:?}, dot={:.4}", message, q1, q2, dot);
}