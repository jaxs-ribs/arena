use render::gpu_types::CylinderGpu;

#[test]
fn debug_cylinder_orientation() {
    println!("\n=== Testing Cylinder Orientation Pipeline ===\n");
    
    // Step 1: Create a test orientation (45 degrees around Y)
    let angle = std::f32::consts::PI / 4.0;
    let half_angle = angle / 2.0;
    let test_orientation = [
        0.0,                // x component
        half_angle.sin(),   // y component (for Y-axis rotation)
        0.0,                // z component
        half_angle.cos(),   // w component
    ];
    
    println!("1. Created test orientation (45° Y rotation):");
    println!("   Quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             test_orientation[0], test_orientation[1], 
             test_orientation[2], test_orientation[3]);
    
    // Step 2: Create a mock cylinder with GPU struct directly
    let gpu_cylinder = CylinderGpu {
        orientation: test_orientation,
        center: [0.0, 0.0, 0.0, 0.0],
        radius: 0.5,
        half_height: 1.0,
        color: [1.0, 0.0, 0.0, 1.0],
        _pad: [0.0; 2],
    };
    
    println!("\n2. Created CylinderGpu:");
    println!("   orientation: {:?}", gpu_cylinder.orientation);
    println!("   center: [{:.1}, {:.1}, {:.1}]", 
             gpu_cylinder.center[0], gpu_cylinder.center[1], gpu_cylinder.center[2]);
    println!("   radius: {}", gpu_cylinder.radius);
    println!("   half_height: {}", gpu_cylinder.half_height);
    
    // Step 3: Check what would be sent to the shader
    let gpu_data = bytemuck::bytes_of(&gpu_cylinder);
    println!("\n3. Raw GPU buffer data (48 bytes total):");
    println!("   Bytes 0-15 (orientation): {:?}", &gpu_data[0..16]);
    println!("   Bytes 16-31 (center): {:?}", &gpu_data[16..32]);
    println!("   Bytes 32-35 (radius): {:?}", &gpu_data[32..36]);
    println!("   Bytes 36-39 (half_height): {:?}", &gpu_data[36..40]);
    
    // Step 4: Verify the orientation values in the buffer
    let orientation_bytes = &gpu_data[0..16];
    for i in 0..4 {
        let offset = i * 4;
        let value = f32::from_ne_bytes([
            orientation_bytes[offset],
            orientation_bytes[offset + 1],
            orientation_bytes[offset + 2],
            orientation_bytes[offset + 3],
        ]);
        println!("   Orientation[{}] = {:.3}", i, value);
        
        // Verify it matches our test orientation
        assert!((value - test_orientation[i]).abs() < 0.001, 
                "Orientation component {} doesn't match", i);
    }
    
    println!("\n✅ Orientation correctly stored in GPU buffer!");
    
    // Step 5: Calculate expected rotation matrix
    println!("\n4. Expected rotation matrix from quaternion:");
    let x = test_orientation[0];
    let y = test_orientation[1];
    let z = test_orientation[2];
    let w = test_orientation[3];
    
    // Manual quaternion to matrix conversion
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    
    let m00 = 1.0 - 2.0 * (yy + zz);
    let m01 = 2.0 * (xy + wz);
    let m02 = 2.0 * (xz - wy);
    
    let m10 = 2.0 * (xy - wz);
    let m11 = 1.0 - 2.0 * (xx + zz);
    let m12 = 2.0 * (yz + wx);
    
    let m20 = 2.0 * (xz + wy);
    let m21 = 2.0 * (yz - wx);
    let m22 = 1.0 - 2.0 * (xx + yy);
    
    println!("   [{:.3}, {:.3}, {:.3}]", m00, m01, m02);
    println!("   [{:.3}, {:.3}, {:.3}]", m10, m11, m12);
    println!("   [{:.3}, {:.3}, {:.3}]", m20, m21, m22);
    
    // For 45° Y rotation, we expect:
    // [0.707,  0,  0.707]
    // [0,      1,  0    ]
    // [-0.707, 0,  0.707]
    let expected = 0.707;
    assert!((m00 - expected).abs() < 0.01, "m00 should be ~0.707");
    assert!((m02 - expected).abs() < 0.01, "m02 should be ~0.707");
    assert!((m20 - (-expected)).abs() < 0.01, "m20 should be ~-0.707");
    assert!((m22 - expected).abs() < 0.01, "m22 should be ~0.707");
    
    println!("\n✅ Rotation matrix is correct for 45° Y rotation!");
    println!("\n⚠️  If cylinders still appear unrotated, the issue is in the shader!");
}