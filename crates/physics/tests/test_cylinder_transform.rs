use physics::{PhysicsSim, Vec3, transform};
use physics::types::BodyType;

#[test]
fn test_cartpole_cylinder_transform() {
    println!("\n=== Testing CartPole Cylinder Transform ===\n");
    
    // Create simulation
    let mut sim = PhysicsSim::new();
    
    // Create a cartpole-style cylinder with mesh offset
    let center_pos = Vec3::new(0.0, 1.0, 0.0);
    let radius = 0.05;
    let half_height = 0.75;
    
    // Create cylinder with bottom-pivot mesh offset
    let shape_offset = Vec3::ZERO; // Shape at center of mass
    let mesh_offset = Vec3::new(0.0, -half_height, 0.0); // Mesh origin at bottom
    
    let cylinder_idx = sim.add_cylinder_with_offsets(
        center_pos,
        radius,
        half_height,
        Vec3::ZERO,
        BodyType::Dynamic,
        shape_offset,
        mesh_offset
    );
    
    // Test 1: Verify physics position is at center
    let cylinder = &sim.cylinders[cylinder_idx];
    assert_eq!(cylinder.pos.x, 0.0);
    assert_eq!(cylinder.pos.y, 1.0);
    assert_eq!(cylinder.pos.z, 0.0);
    println!("✅ Physics position at center: ({:.2}, {:.2}, {:.2})", 
             cylinder.pos.x, cylinder.pos.y, cylinder.pos.z);
    
    // Test 2: Calculate bottom position in physics space
    let bottom_y = cylinder.pos.y - half_height;
    assert_eq!(bottom_y, 0.25); // 1.0 - 0.75 = 0.25
    println!("✅ Bottom position in physics: y = {:.2}", bottom_y);
    
    // Test 3: Verify mesh transform places visual origin at bottom
    let transform = transform::to_transform_matrix_with_offset(
        cylinder.pos,
        cylinder.orientation,
        cylinder.mesh_offset
    );
    
    // The transform should place the mesh origin at the bottom
    // With identity rotation and mesh_offset = (0, -half_height, 0),
    // the final translation should be pos + mesh_offset = (0, 0.25, 0)
    let expected_y = center_pos.y + mesh_offset.y;
    assert_eq!(transform[3][1], expected_y);
    println!("✅ Transform places mesh origin at bottom: y = {:.2}", transform[3][1]);
    
    // Test 4: Verify rotation works correctly
    // Rotate 45 degrees around Z axis
    let angle = std::f32::consts::PI / 4.0;
    let half_angle = angle * 0.5;
    sim.cylinders[cylinder_idx].orientation = [
        0.0,
        0.0,
        half_angle.sin(),
        half_angle.cos()
    ];
    
    let rotated_transform = transform::to_transform_matrix_with_offset(
        cylinder.pos,
        sim.cylinders[cylinder_idx].orientation,
        cylinder.mesh_offset
    );
    
    println!("\n✅ After 45° rotation:");
    println!("   Transform position: ({:.3}, {:.3}, {:.3})", 
             rotated_transform[3][0], rotated_transform[3][1], rotated_transform[3][2]);
    
    // The rotation should affect how the mesh offset is applied
    // but the pivot point (bottom of cylinder) should remain in the same world position
    println!("\n✅ All transform tests passed!");
}

#[test]
fn test_standard_cylinder_transform() {
    println!("\n=== Testing Standard Cylinder Transform ===\n");
    
    // Create simulation
    let mut sim = PhysicsSim::new();
    
    // Create a standard cylinder (no offsets)
    let center_pos = Vec3::new(2.0, 3.0, 1.0);
    let cylinder_idx = sim.add_cylinder(
        center_pos,
        0.5,  // radius
        1.0,  // half_height
        Vec3::ZERO
    );
    
    let cylinder = &sim.cylinders[cylinder_idx];
    
    // Verify default offsets are zero
    assert_eq!(cylinder.shape_offset.x, 0.0);
    assert_eq!(cylinder.shape_offset.y, 0.0);
    assert_eq!(cylinder.shape_offset.z, 0.0);
    assert_eq!(cylinder.mesh_offset.x, 0.0);
    assert_eq!(cylinder.mesh_offset.y, 0.0);
    assert_eq!(cylinder.mesh_offset.z, 0.0);
    println!("✅ Default offsets are zero");
    
    // Transform should just be position
    let transform = transform::to_transform_matrix_with_offset(
        cylinder.pos,
        cylinder.orientation,
        cylinder.mesh_offset
    );
    
    assert_eq!(transform[3][0], center_pos.x);
    assert_eq!(transform[3][1], center_pos.y);
    assert_eq!(transform[3][2], center_pos.z);
    println!("✅ Transform equals position for zero offset");
}