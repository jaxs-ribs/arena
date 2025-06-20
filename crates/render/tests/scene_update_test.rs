use render::{Scene, SceneObject, Transform, Camera, Light, LightType};
use glam::{Vec3, Quat, Mat4, EulerRot};

#[test]
fn test_scene_update_preserves_orientation() {
    println!("\n=== Testing Scene Update Orientation Preservation ===");
    
    // Create a scene
    let mut scene = Scene::new();
    
    // Test various rotations
    let test_cases = vec![
        ("45° around X", Quat::from_axis_angle(Vec3::X, std::f32::consts::PI / 4.0)),
        ("90° around Y", Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 2.0)),
        ("30° around Z", Quat::from_axis_angle(Vec3::Z, std::f32::consts::PI / 6.0)),
        ("Complex rotation", Quat::from_euler(glam::EulerRot::XYZ, 0.5, 0.7, 0.3)),
    ];
    
    for (desc, rotation) in test_cases {
        println!("\n1. Testing {}", desc);
        println!("   Input rotation: {:?}", rotation);
        println!("   Rotation angle: {} degrees", rotation.angle().to_degrees());
        
        // Clear scene
        scene = Scene::new();
        
        // Add object with specific rotation
        let transform = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation,
            scale: Vec3::new(1.0, 1.0, 1.0),
        };
        
        // Create a dummy mesh index
        let mesh_index = 0;
        let material_index = 0;
        
        let object = SceneObject::new(transform, mesh_index, material_index);
        
        // Verify the object stores the correct transform
        println!("   Object transform rotation: {:?}", object.transform.rotation);
        assert_quaternions_equal(
            rotation,
            object.transform.rotation,
            "Object should store the correct rotation"
        );
        
        // Add to scene
        scene.add_object(object);
        
        // Get objects back from scene
        let objects = scene.get_objects();
        assert_eq!(objects.len(), 1, "Scene should have one object");
        
        let retrieved_object = &objects[0];
        println!("   Retrieved object rotation: {:?}", retrieved_object.transform.rotation);
        
        // Verify rotation is preserved
        assert_quaternions_equal(
            rotation,
            retrieved_object.transform.rotation,
            "Scene should preserve object rotation"
        );
        
        // Test transform matrix generation
        let matrix = Mat4::from_scale_rotation_translation(
            transform.scale,
            transform.rotation,
            transform.position,
        );
        
        println!("   Transform matrix:");
        print_matrix(&matrix);
        
        // Extract rotation back from matrix
        let extracted = extract_rotation_from_matrix(&matrix);
        println!("   Extracted rotation from matrix: {:?}", extracted);
        
        assert_quaternions_equal(
            rotation,
            extracted,
            "Matrix should preserve rotation"
        );
    }
    
    // Test with multiple objects
    println!("\n2. Testing multiple objects in scene");
    scene = Scene::new();
    
    let rotations = vec![
        Quat::from_axis_angle(Vec3::X, 0.5),
        Quat::from_axis_angle(Vec3::Y, 1.0),
        Quat::from_axis_angle(Vec3::Z, 1.5),
    ];
    
    for (i, rot) in rotations.iter().enumerate() {
        let transform = Transform {
            position: Vec3::new(i as f32, 0.0, 0.0),
            rotation: *rot,
            scale: Vec3::ONE,
        };
        
        scene.add_object(SceneObject::new(transform, 0, 0));
    }
    
    // Verify all objects maintain their rotations
    let objects = scene.get_objects();
    for (i, obj) in objects.iter().enumerate() {
        println!("   Object {} rotation: {:?}", i, obj.transform.rotation);
        assert_quaternions_equal(
            rotations[i],
            obj.transform.rotation,
            &format!("Object {} should maintain its rotation", i)
        );
    }
    
    // Test camera and light (ensure they don't affect object transforms)
    println!("\n3. Testing camera and light additions");
    
    scene.set_camera(Camera::new(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
        45.0,
        1.0,
        0.1,
        100.0,
    ));
    
    scene.add_light(Light::new(
        Vec3::new(10.0, 10.0, 10.0),
        Vec3::ONE,
        1.0,
        LightType::Point { range: 50.0 },
    ));
    
    // Verify objects still have correct rotations
    let objects_after = scene.get_objects();
    for (i, obj) in objects_after.iter().enumerate() {
        assert_quaternions_equal(
            rotations[i],
            obj.transform.rotation,
            &format!("Object {} should maintain rotation after camera/light addition", i)
        );
    }
    
    println!("\n=== Scene Update Test Completed Successfully ===");
}

#[test]
fn test_transform_identity_check() {
    println!("\n=== Testing Transform Identity Detection ===");
    
    // Test that we can detect identity vs non-identity rotations
    let identity = Quat::IDENTITY;
    let not_identity = Quat::from_axis_angle(Vec3::Y, 0.1); // Small rotation
    
    println!("Identity quaternion: {:?}", identity);
    println!("Non-identity quaternion: {:?}", not_identity);
    
    assert!(
        identity.w > 0.99,
        "Identity quaternion should have w close to 1.0"
    );
    
    assert!(
        not_identity.w < 0.99,
        "Non-identity quaternion should have w less than 0.99"
    );
    
    // Test that default Transform has identity rotation
    let default_transform = Transform {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    
    assert_eq!(
        default_transform.rotation,
        Quat::IDENTITY,
        "Default transform should have identity rotation"
    );
}

fn print_matrix(mat: &Mat4) {
    let cols = mat.to_cols_array_2d();
    for row in 0..4 {
        println!("      [{:.3}, {:.3}, {:.3}, {:.3}]",
            cols[0][row], cols[1][row], cols[2][row], cols[3][row]
        );
    }
}

fn extract_rotation_from_matrix(mat: &Mat4) -> Quat {
    // Use glam's built-in decomposition
    let (scale, rotation, translation) = mat.to_scale_rotation_translation();
    rotation
}

fn assert_quaternions_equal(q1: Quat, q2: Quat, message: &str) {
    let tolerance = 0.001;
    
    // Check if quaternions are approximately equal (considering q and -q are the same rotation)
    let dot = q1.dot(q2);
    let same = dot.abs() > 1.0 - tolerance;
    
    assert!(same, "{}: q1={:?}, q2={:?}, dot={}", message, q1, q2, dot);
}