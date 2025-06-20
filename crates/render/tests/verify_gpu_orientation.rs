use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CylinderGpu {
    pos: [f32; 3],
    radius: f32,
    height: f32,
    _pad0: [f32; 3],
    orientation: [f32; 4],
}

#[test]
fn verify_gpu_orientation_format() {
    println!("\n=== Verifying GPU Orientation Format ===");
    
    // Create test cylinder with known orientation
    let test_cylinder = CylinderGpu {
        pos: [0.0, 1.5, 0.0],
        radius: 0.05,
        height: 1.0,
        _pad0: [0.0; 3],
        orientation: [0.0, 0.0, 0.259, 0.966], // 30-degree rotation around Z
    };
    
    // Verify the struct size and alignment
    println!("Cylinder GPU struct size: {} bytes", std::mem::size_of::<CylinderGpu>());
    println!("Expected size: 48 bytes (12 floats * 4 bytes)");
    assert_eq!(std::mem::size_of::<CylinderGpu>(), 48);
    
    // Verify byte layout
    let bytes = bytemuck::bytes_of(&test_cylinder);
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    
    println!("\nFloat layout:");
    println!("  Position: [{:.3}, {:.3}, {:.3}]", floats[0], floats[1], floats[2]);
    println!("  Radius: {:.3}", floats[3]);
    println!("  Height: {:.3}", floats[4]);
    println!("  Padding: [{:.3}, {:.3}, {:.3}]", floats[5], floats[6], floats[7]);
    println!("  Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", floats[8], floats[9], floats[10], floats[11]);
    
    // Verify orientation is at the correct offset
    assert_eq!(floats[8], test_cylinder.orientation[0], "X component");
    assert_eq!(floats[9], test_cylinder.orientation[1], "Y component");
    assert_eq!(floats[10], test_cylinder.orientation[2], "Z component");
    assert_eq!(floats[11], test_cylinder.orientation[3], "W component");
    
    println!("\n✓ GPU orientation format is correct");
}

#[test]
fn verify_cylinder_array_upload() {
    println!("\n=== Verifying Cylinder Array Upload ===");
    
    // Create multiple cylinders with different orientations
    let cylinders = vec![
        CylinderGpu {
            pos: [0.0, 1.0, 0.0],
            radius: 0.1,
            height: 2.0,
            _pad0: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0], // Identity
        },
        CylinderGpu {
            pos: [2.0, 1.0, 0.0],
            radius: 0.1,
            height: 2.0,
            _pad0: [0.0; 3],
            orientation: [0.0, 0.0, 0.259, 0.966], // 30 degrees
        },
        CylinderGpu {
            pos: [-2.0, 1.0, 0.0],
            radius: 0.1,
            height: 2.0,
            _pad0: [0.0; 3],
            orientation: [0.0, 0.0, 0.707, 0.707], // 90 degrees
        },
    ];
    
    // Convert to bytes
    let bytes: &[u8] = bytemuck::cast_slice(&cylinders);
    let total_size = bytes.len();
    
    println!("Total byte size for {} cylinders: {} bytes", cylinders.len(), total_size);
    println!("Expected: {} bytes", cylinders.len() * 48);
    assert_eq!(total_size, cylinders.len() * 48);
    
    // Verify each cylinder's orientation in the byte stream
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    for (i, cylinder) in cylinders.iter().enumerate() {
        let offset = i * 12; // 12 floats per cylinder
        let orientation_offset = offset + 8; // Orientation starts at float 8
        
        println!("\nCylinder {} orientation:", i);
        println!("  Expected: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 cylinder.orientation[0], cylinder.orientation[1], 
                 cylinder.orientation[2], cylinder.orientation[3]);
        println!("  In buffer: [{:.3}, {:.3}, {:.3}, {:.3}]",
                 floats[orientation_offset], floats[orientation_offset + 1],
                 floats[orientation_offset + 2], floats[orientation_offset + 3]);
        
        assert_eq!(floats[orientation_offset], cylinder.orientation[0]);
        assert_eq!(floats[orientation_offset + 1], cylinder.orientation[1]);
        assert_eq!(floats[orientation_offset + 2], cylinder.orientation[2]);
        assert_eq!(floats[orientation_offset + 3], cylinder.orientation[3]);
    }
    
    println!("\n✓ Cylinder array upload format is correct");
}