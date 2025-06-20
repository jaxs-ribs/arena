use glam::{Vec3, Quat, Mat4};
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// Define Transform locally since we're testing it
#[derive(Clone, Copy, Debug)]
struct Transform {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TransformUniform {
    model: [[f32; 4]; 4],
}

#[test]
fn test_transform_buffer_update() {
    println!("\n=== Testing Transform Buffer Update ===");
    
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    runtime.block_on(async {
        // Create test device and queue
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
        
        // Test 1: Direct transform buffer creation
        println!("\n1. Testing direct transform buffer creation...");
        
        let test_rotation = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 3.0); // 60 degrees
        let test_position = Vec3::new(1.0, 2.0, 3.0);
        let test_scale = Vec3::new(1.0, 1.0, 1.0);
        
        let transform_matrix = Mat4::from_scale_rotation_translation(
            test_scale,
            test_rotation,
            test_position,
        );
        
        println!("   Test rotation: {:?}", test_rotation);
        println!("   Test rotation angle: {} degrees", test_rotation.angle().to_degrees());
        println!("   Transform matrix:");
        print_matrix(&transform_matrix);
        
        // Create transform uniform
        let transform_uniform = TransformUniform {
            model: transform_matrix.to_cols_array_2d(),
        };
        
        // Verify the data before sending to GPU
        let matrix_from_uniform = Mat4::from_cols_array_2d(&transform_uniform.model);
        println!("\n   Matrix from uniform data:");
        print_matrix(&matrix_from_uniform);
        
        // Test 2: Verify byte representation
        println!("\n2. Testing byte representation...");
        let bytes = bytemuck::cast_slice(&[transform_uniform]);
        println!("   Transform uniform size: {} bytes", bytes.len());
        println!("   First few floats: {:?}", &bytes[0..16]);
        
        // Reconstruct matrix from bytes
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        let reconstructed = Mat4::from_cols_array(&[
            floats[0], floats[1], floats[2], floats[3],
            floats[4], floats[5], floats[6], floats[7],
            floats[8], floats[9], floats[10], floats[11],
            floats[12], floats[13], floats[14], floats[15],
        ]);
        
        println!("\n   Reconstructed matrix from bytes:");
        print_matrix(&reconstructed);
        
        // Verify rotation is preserved
        let extracted_rotation = extract_rotation_from_matrix(&reconstructed);
        println!("\n   Extracted rotation from reconstructed matrix: {:?}", extracted_rotation);
        
        assert_quaternions_equal(
            test_rotation,
            extracted_rotation,
            "Rotation should be preserved through byte conversion"
        );
        
        // Test 3: Create actual GPU buffer and verify
        println!("\n3. Testing GPU buffer creation and update...");
        
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Transform Buffer"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        });
        
        println!("   GPU buffer created successfully");
        println!("   Buffer size: {} bytes", buffer.size());
        
        // Test 4: Multiple transforms
        println!("\n4. Testing multiple transforms...");
        
        let transforms = vec![
            Transform {
                position: Vec3::new(0.0, 0.0, 0.0),
                rotation: Quat::from_axis_angle(Vec3::X, std::f32::consts::PI / 6.0), // 30 deg
                scale: Vec3::ONE,
            },
            Transform {
                position: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 4.0), // 45 deg
                scale: Vec3::ONE,
            },
            Transform {
                position: Vec3::new(-1.0, 0.0, 1.0),
                rotation: Quat::from_axis_angle(Vec3::Z, std::f32::consts::PI / 2.0), // 90 deg
                scale: Vec3::new(2.0, 2.0, 2.0),
            },
        ];
        
        let mut uniforms = Vec::new();
        for (i, transform) in transforms.iter().enumerate() {
            println!("\n   Transform {}: rotation = {:?} ({} deg)",
                i,
                transform.rotation,
                transform.rotation.angle().to_degrees()
            );
            
            let matrix = Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.position,
            );
            
            uniforms.push(TransformUniform {
                model: matrix.to_cols_array_2d(),
            });
        }
        
        let multi_bytes = bytemuck::cast_slice(&uniforms);
        let multi_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Multi Transform Buffer"),
            contents: multi_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        });
        
        println!("\n   Created buffer with {} transforms", transforms.len());
        println!("   Total buffer size: {} bytes", multi_buffer.size());
        
        println!("\n=== Transform Buffer Test Completed Successfully ===");
    });
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
    // Extract the rotation from the transformation matrix
    // Note: This assumes the matrix has uniform scaling or no scaling
    let m = mat.to_cols_array_2d();
    
    // Extract the 3x3 rotation part
    let trace = m[0][0] + m[1][1] + m[2][2];
    
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m[2][1] - m[1][2]) * s;
        let y = (m[0][2] - m[2][0]) * s;
        let z = (m[1][0] - m[0][1]) * s;
        Quat::from_xyzw(x, y, z, w).normalize()
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        Quat::from_xyzw(x, y, z, w).normalize()
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        Quat::from_xyzw(x, y, z, w).normalize()
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        Quat::from_xyzw(x, y, z, w).normalize()
    }
}

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
    
    assert!(same || opposite, "{}: q1={:?}, q2={:?}", message, q1, q2);
}