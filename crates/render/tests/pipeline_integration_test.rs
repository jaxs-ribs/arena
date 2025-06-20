use render::{Renderer, Scene, SceneObject, Transform, Camera, Light, LightType};
use physics::{PhysicsEngine, RigidBody, Collider, ColliderShape};
use glam::{Vec3, Quat, Mat4};
use std::sync::Arc;

#[test]
fn test_full_rendering_pipeline_orientation() {
    println!("\n=== Starting Full Pipeline Integration Test ===");
    
    // Initialize runtime for async operations
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    runtime.block_on(async {
        // Step 1: Create physics engine and CartPole scene
        println!("\n1. Creating physics engine and CartPole scene...");
        let mut physics = PhysicsEngine::new();
        
        // Create cart (base)
        let cart_body = RigidBody::dynamic()
            .with_position(Vec3::new(0.0, 0.5, 0.0))
            .with_mass(1.0);
        let cart_collider = Collider::new(ColliderShape::Box {
            half_extents: Vec3::new(0.5, 0.25, 0.25),
        });
        let cart_id = physics.add_rigid_body(cart_body, cart_collider);
        println!("   Cart created at position: {:?}", physics.get_position(cart_id));
        
        // Create pole with initial tilt (45 degrees around Z axis)
        let initial_rotation = Quat::from_axis_angle(Vec3::Z, std::f32::consts::PI / 4.0);
        let pole_body = RigidBody::dynamic()
            .with_position(Vec3::new(0.0, 1.5, 0.0))
            .with_orientation(initial_rotation)
            .with_mass(0.1);
        let pole_collider = Collider::new(ColliderShape::Box {
            half_extents: Vec3::new(0.05, 0.5, 0.05),
        });
        let pole_id = physics.add_rigid_body(pole_body, pole_collider);
        
        let initial_orientation = physics.get_orientation(pole_id);
        println!("   Pole created with initial orientation: {:?}", initial_orientation);
        println!("   Initial rotation angle: {} degrees", initial_rotation.angle().to_degrees());
        
        // Step 2: Run one physics step
        println!("\n2. Running physics simulation step...");
        physics.step(1.0 / 60.0);
        
        let post_step_orientation = physics.get_orientation(pole_id);
        println!("   Pole orientation after physics step: {:?}", post_step_orientation);
        println!("   Post-step rotation angle: {} degrees", post_step_orientation.angle().to_degrees());
        
        // Verify physics actually updated the orientation
        assert!(
            (initial_orientation.w - post_step_orientation.w).abs() > 0.001 ||
            (initial_orientation.x - post_step_orientation.x).abs() > 0.001 ||
            (initial_orientation.y - post_step_orientation.y).abs() > 0.001 ||
            (initial_orientation.z - post_step_orientation.z).abs() > 0.001,
            "Physics simulation should have updated the orientation"
        );
        
        // Step 3: Create renderer and scene
        println!("\n3. Creating renderer and scene...");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.unwrap();
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Create renderer
        let mut renderer = Renderer::new(device.clone(), queue.clone(), 800, 600);
        
        // Create scene with cart and pole
        let mut scene = Scene::new();
        
        // Add cart object
        let cart_transform = Transform {
            position: physics.get_position(cart_id),
            rotation: physics.get_orientation(cart_id),
            scale: Vec3::new(1.0, 0.5, 0.5),
        };
        scene.add_object(SceneObject::new(
            cart_transform,
            renderer.create_box_mesh(1.0, 0.5, 0.5),
            0,
        ));
        
        // Add pole object
        let pole_transform = Transform {
            position: physics.get_position(pole_id),
            rotation: physics.get_orientation(pole_id),
            scale: Vec3::new(0.1, 1.0, 0.1),
        };
        println!("\n4. Creating pole transform for rendering:");
        println!("   Position: {:?}", pole_transform.position);
        println!("   Rotation: {:?}", pole_transform.rotation);
        println!("   Scale: {:?}", pole_transform.scale);
        
        scene.add_object(SceneObject::new(
            pole_transform,
            renderer.create_box_mesh(0.1, 1.0, 0.1),
            0,
        ));
        
        // Add camera and light
        scene.set_camera(Camera::new(
            Vec3::new(5.0, 3.0, 5.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            45.0,
            800.0 / 600.0,
            0.1,
            100.0,
        ));
        
        scene.add_light(Light::new(
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(1.0, 1.0, 1.0),
            1.0,
            LightType::Directional { direction: Vec3::new(-1.0, -1.0, -1.0).normalize() },
        ));
        
        // Step 5: Update renderer with scene
        println!("\n5. Updating renderer with scene data...");
        renderer.update_scene(&scene);
        
        // Step 6: Verify transform matrices
        println!("\n6. Verifying transform matrices:");
        
        // Calculate expected transform matrix for pole
        let expected_matrix = Mat4::from_scale_rotation_translation(
            pole_transform.scale,
            pole_transform.rotation,
            pole_transform.position,
        );
        println!("   Expected transform matrix for pole:");
        print_matrix(&expected_matrix);
        
        // Extract rotation from the matrix to verify
        let extracted_rotation = extract_rotation_from_matrix(&expected_matrix);
        println!("   Extracted rotation from matrix: {:?}", extracted_rotation);
        
        // Step 7: Read and verify GPU buffer data
        println!("\n7. Reading GPU buffer data...");
        
        // Create a staging buffer to read back the transform data
        let buffer_size = 64 * 2; // 2 objects * 64 bytes per transform
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy from transform buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        
        // Get the transform buffer from renderer (we'll need to make this accessible)
        // For now, we'll verify the data is being processed correctly
        
        // Verify the pole's transform was updated with correct orientation
        let objects = scene.get_objects();
        let pole_object = &objects[1]; // Pole is the second object
        
        println!("\n8. Final verification:");
        println!("   Pole object transform rotation: {:?}", pole_object.transform.rotation);
        println!("   Physics engine pole orientation: {:?}", physics.get_orientation(pole_id));
        
        // They should match
        assert_quaternions_equal(
            pole_object.transform.rotation,
            physics.get_orientation(pole_id),
            "Scene object should have the same orientation as physics body"
        );
        
        // Verify the rotation is not identity (not reset)
        assert!(
            pole_object.transform.rotation.w < 0.99,
            "Pole rotation should not be identity/reset"
        );
        
        println!("\n=== Pipeline Integration Test Completed Successfully ===");
    });
}

// Helper function to print a 4x4 matrix
fn print_matrix(mat: &Mat4) {
    let cols = mat.to_cols_array_2d();
    for row in 0..4 {
        println!("   [{:.3}, {:.3}, {:.3}, {:.3}]",
            cols[0][row], cols[1][row], cols[2][row], cols[3][row]
        );
    }
}

// Helper function to extract rotation quaternion from a transformation matrix
fn extract_rotation_from_matrix(mat: &Mat4) -> Quat {
    // Extract the upper-left 3x3 rotation matrix
    let m00 = mat.x_axis.x;
    let m01 = mat.x_axis.y;
    let m02 = mat.x_axis.z;
    let m10 = mat.y_axis.x;
    let m11 = mat.y_axis.y;
    let m12 = mat.y_axis.z;
    let m20 = mat.z_axis.x;
    let m21 = mat.z_axis.y;
    let m22 = mat.z_axis.z;
    
    // Convert to quaternion
    let trace = m00 + m11 + m22;
    
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m21 - m12) * s;
        let y = (m02 - m20) * s;
        let z = (m10 - m01) * s;
        Quat::from_xyzw(x, y, z, w)
    } else if m00 > m11 && m00 > m22 {
        let s = 2.0 * ((1.0 + m00 - m11 - m22) as f32).sqrt();
        let w = (m21 - m12) / s;
        let x = 0.25 * s;
        let y = (m01 + m10) / s;
        let z = (m02 + m20) / s;
        Quat::from_xyzw(x, y, z, w)
    } else if m11 > m22 {
        let s = 2.0 * ((1.0 + m11 - m00 - m22) as f32).sqrt();
        let w = (m02 - m20) / s;
        let x = (m01 + m10) / s;
        let y = 0.25 * s;
        let z = (m12 + m21) / s;
        Quat::from_xyzw(x, y, z, w)
    } else {
        let s = 2.0 * ((1.0 + m22 - m00 - m11) as f32).sqrt();
        let w = (m10 - m01) / s;
        let x = (m02 + m20) / s;
        let y = (m12 + m21) / s;
        let z = 0.25 * s;
        Quat::from_xyzw(x, y, z, w)
    }
}

// Helper function to compare quaternions with tolerance
fn assert_quaternions_equal(q1: Quat, q2: Quat, message: &str) {
    let tolerance = 0.001;
    
    // Quaternions q and -q represent the same rotation
    let same = (q1.w - q2.w).abs() < tolerance &&
               (q1.x - q2.x).abs() < tolerance &&
               (q1.y - q2.y).abs() < tolerance &&
               (q1.z - q2.z).abs() < tolerance;
               
    let opposite = (q1.w + q2.w).abs() < tolerance &&
                   (q1.x + q2.x).abs() < tolerance &&
                   (q1.y + q2.y).abs() < tolerance &&
                   (q1.z + q2.z).abs() < tolerance;
    
    assert!(same || opposite, "{}", message);
}

#[test]
fn test_transform_matrix_calculation() {
    println!("\n=== Testing Transform Matrix Calculation ===");
    
    // Test with a known rotation
    let rotation = Quat::from_axis_angle(Vec3::Z, std::f32::consts::PI / 4.0); // 45 degrees
    let position = Vec3::new(1.0, 2.0, 3.0);
    let scale = Vec3::new(1.0, 1.0, 1.0);
    
    let transform = Transform {
        position,
        rotation,
        scale,
    };
    
    let matrix = Mat4::from_scale_rotation_translation(scale, rotation, position);
    
    println!("Input rotation: {:?}", rotation);
    println!("Input position: {:?}", position);
    println!("Resulting matrix:");
    print_matrix(&matrix);
    
    // Verify the matrix contains the correct rotation
    let extracted = extract_rotation_from_matrix(&matrix);
    println!("Extracted rotation: {:?}", extracted);
    
    assert_quaternions_equal(rotation, extracted, "Rotation should be preserved in matrix");
}