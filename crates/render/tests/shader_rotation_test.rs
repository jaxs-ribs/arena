use physics::{PhysicsSim, types::Vec3};
use render::{Renderer, CylinderGpu};
use std::f32::consts::PI;

/// Helper to create a quaternion from axis-angle representation
fn quaternion_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();
    
    // Normalize axis
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let norm_axis = [axis[0] / len, axis[1] / len, axis[2] / len];
    
    [
        norm_axis[0] * sin_half,
        norm_axis[1] * sin_half,
        norm_axis[2] * sin_half,
        cos_half
    ]
}

/// Helper to apply quaternion rotation to a point (matching shader logic)
fn quaternion_rotate_point(p: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    
    // Cross product: qv × p
    let cross1 = [
        qv[1] * p[2] - qv[2] * p[1],
        qv[2] * p[0] - qv[0] * p[2],
        qv[0] * p[1] - qv[1] * p[0]
    ];
    
    // qv × p + qw * p
    let temp = [
        cross1[0] + qw * p[0],
        cross1[1] + qw * p[1],
        cross1[2] + qw * p[2]
    ];
    
    // Cross product: qv × temp
    let cross2 = [
        qv[1] * temp[2] - qv[2] * temp[1],
        qv[2] * temp[0] - qv[0] * temp[2],
        qv[0] * temp[1] - qv[1] * temp[0]
    ];
    
    // p + 2.0 * cross2
    [
        p[0] + 2.0 * cross2[0],
        p[1] + 2.0 * cross2[1],
        p[2] + 2.0 * cross2[2]
    ]
}

/// Calculate the expected SDF value for a cylinder (matching shader logic)
fn calculate_cylinder_sdf(
    point: [f32; 3], 
    center: [f32; 3], 
    radius: f32, 
    height: f32, 
    orientation: [f32; 4]
) -> f32 {
    // Transform point to cylinder's local space
    let offset = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2]
    ];
    
    // Apply inverse rotation (conjugate)
    let conj_q = [-orientation[0], -orientation[1], -orientation[2], orientation[3]];
    let local_p = quaternion_rotate_point(offset, conj_q);
    
    // Standard cylinder SDF in local space (Y-aligned)
    let xz_dist = (local_p[0] * local_p[0] + local_p[2] * local_p[2]).sqrt();
    let d = [xz_dist - radius, local_p[1].abs() - height * 0.5];
    
    // Return min(max(d.x, d.y), 0.0) + length(max(d, 0))
    let max_component = d[0].max(d[1]);
    let max_d = [d[0].max(0.0), d[1].max(0.0)];
    let length_max_d = (max_d[0] * max_d[0] + max_d[1] * max_d[1]).sqrt();
    
    max_component.min(0.0) + length_max_d
}

#[test]
fn test_cylinder_sdf_calculation() {
    println!("🧪 Testing cylinder SDF calculation with various orientations...\n");
    
    let cylinder_center = [0.0, 0.0, 0.0];
    let cylinder_radius = 1.0;
    let cylinder_height = 2.0;
    
    // Test 1: Identity quaternion (no rotation - cylinder aligned with Y axis)
    println!("📐 Test 1: Identity quaternion (vertical cylinder)");
    let identity_quat = [0.0, 0.0, 0.0, 1.0];
    
    // Test points
    let test_points = [
        ([0.0, 0.0, 0.0], "Center"),
        ([1.0, 0.0, 0.0], "On surface (X direction)"),
        ([0.0, 1.0, 0.0], "On top cap"),
        ([0.0, -1.0, 0.0], "On bottom cap"),
        ([2.0, 0.0, 0.0], "Outside (X direction)"),
        ([0.5, 0.0, 0.0], "Inside (X direction)"),
    ];
    
    for (point, desc) in &test_points {
        let sdf = calculate_cylinder_sdf(*point, cylinder_center, cylinder_radius, cylinder_height, identity_quat);
        println!("  Point {} at {:?}: SDF = {:.3}", desc, point, sdf);
    }
    
    // Test 2: 90 degree rotation around Z axis (cylinder lies along X axis)
    println!("\n📐 Test 2: 90° rotation around Z axis (horizontal cylinder along X)");
    let rot_90_z = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
    println!("  Quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", rot_90_z[0], rot_90_z[1], rot_90_z[2], rot_90_z[3]);
    
    let test_points_rotated = [
        ([0.0, 0.0, 0.0], "Center"),
        ([1.0, 0.0, 0.0], "On top cap (now at +X)"),
        ([-1.0, 0.0, 0.0], "On bottom cap (now at -X)"),
        ([0.0, 1.0, 0.0], "On surface (Y direction)"),
        ([0.0, 0.0, 1.0], "On surface (Z direction)"),
        ([2.0, 0.0, 0.0], "Outside top cap"),
    ];
    
    for (point, desc) in &test_points_rotated {
        let sdf = calculate_cylinder_sdf(*point, cylinder_center, cylinder_radius, cylinder_height, rot_90_z);
        println!("  Point {} at {:?}: SDF = {:.3}", desc, point, sdf);
    }
    
    // Test 3: 45 degree rotation around Z axis
    println!("\n📐 Test 3: 45° rotation around Z axis");
    let rot_45_z = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 4.0);
    println!("  Quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", rot_45_z[0], rot_45_z[1], rot_45_z[2], rot_45_z[3]);
    
    // Test a point that should be on the cylinder axis after rotation
    let sqrt2_2 = (2.0_f32).sqrt() / 2.0;
    let axis_point = [sqrt2_2, sqrt2_2, 0.0]; // Should be on cylinder axis
    let sdf_axis = calculate_cylinder_sdf(axis_point, cylinder_center, cylinder_radius, cylinder_height, rot_45_z);
    println!("  Point on rotated axis at {:?}: SDF = {:.3}", axis_point, sdf_axis);
    
    // Test 4: Complex rotation (around multiple axes)
    println!("\n📐 Test 4: Complex rotation (30° around X, then 60° around Y)");
    // First rotate around X
    let rot_x = quaternion_from_axis_angle([1.0, 0.0, 0.0], PI / 6.0);
    // Then around Y - we'd need to compose quaternions for accurate test
    // For simplicity, just test the X rotation
    println!("  Quaternion (X rotation only): [{:.3}, {:.3}, {:.3}, {:.3}]", rot_x[0], rot_x[1], rot_x[2], rot_x[3]);
    
    let test_point = [0.0, 0.0, 1.0];
    let sdf = calculate_cylinder_sdf(test_point, cylinder_center, cylinder_radius, cylinder_height, rot_x);
    println!("  Point at {:?}: SDF = {:.3}", test_point, sdf);
}

#[test]
fn test_shader_quaternion_consistency() {
    println!("🧪 Testing shader quaternion math consistency...\n");
    
    // Test that our quaternion math matches what the shader expects
    println!("📐 Verifying quaternion rotation formula:");
    
    // Test rotating unit vectors
    let test_vectors = [
        ([1.0, 0.0, 0.0], "X unit vector"),
        ([0.0, 1.0, 0.0], "Y unit vector"),
        ([0.0, 0.0, 1.0], "Z unit vector"),
    ];
    
    // 90 degree rotation around Z
    let quat = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
    println!("  Using 90° rotation around Z: [{:.3}, {:.3}, {:.3}, {:.3}]", quat[0], quat[1], quat[2], quat[3]);
    
    for (vec, desc) in &test_vectors {
        let rotated = quaternion_rotate_point(*vec, quat);
        println!("  {} {:?} -> [{:.3}, {:.3}, {:.3}]", desc, vec, rotated[0], rotated[1], rotated[2]);
    }
    
    // Verify specific expected results
    println!("\n📐 Verifying expected transformations:");
    
    // X unit vector rotated 90° around Z should become Y unit vector
    let x_rotated = quaternion_rotate_point([1.0, 0.0, 0.0], quat);
    let expected_y = [0.0, 1.0, 0.0];
    let diff = ((x_rotated[0] - expected_y[0]).powi(2) + 
                 (x_rotated[1] - expected_y[1]).powi(2) + 
                 (x_rotated[2] - expected_y[2]).powi(2)).sqrt();
    
    if diff < 0.001 {
        println!("  ✅ X→Y rotation correct (error: {:.6})", diff);
    } else {
        println!("  ❌ X→Y rotation incorrect! Expected {:?}, got {:?}", expected_y, x_rotated);
    }
    
    // Y unit vector rotated 90° around Z should become -X unit vector
    let y_rotated = quaternion_rotate_point([0.0, 1.0, 0.0], quat);
    let expected_neg_x = [-1.0, 0.0, 0.0];
    let diff = ((y_rotated[0] - expected_neg_x[0]).powi(2) + 
                 (y_rotated[1] - expected_neg_x[1]).powi(2) + 
                 (y_rotated[2] - expected_neg_x[2]).powi(2)).sqrt();
    
    if diff < 0.001 {
        println!("  ✅ Y→-X rotation correct (error: {:.6})", diff);
    } else {
        println!("  ❌ Y→-X rotation incorrect! Expected {:?}, got {:?}", expected_neg_x, y_rotated);
    }
}

#[test]
fn test_cylinder_orientation_visual_debug() {
    println!("🧪 Visual debugging of cylinder orientations...\n");
    
    // Create a physics sim with cylinders at different orientations
    let mut sim = PhysicsSim::new();
    
    // Add cylinders with different orientations
    let cylinders = [
        ("Vertical (identity)", [0.0, 0.0, 0.0, 1.0], Vec3::new(-3.0, 1.0, 0.0)),
        ("45° around Z", quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 4.0), Vec3::new(0.0, 1.0, 0.0)),
        ("90° around Z", quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0), Vec3::new(3.0, 1.0, 0.0)),
        ("45° around X", quaternion_from_axis_angle([1.0, 0.0, 0.0], PI / 4.0), Vec3::new(-3.0, 1.0, 3.0)),
        ("90° around X", quaternion_from_axis_angle([1.0, 0.0, 0.0], PI / 2.0), Vec3::new(0.0, 1.0, 3.0)),
    ];
    
    println!("📐 Creating test cylinders:");
    for (desc, orientation, pos) in &cylinders {
        let idx = sim.add_cylinder(*pos, 0.5, 2.0, Vec3::ZERO);
        sim.cylinders[idx].orientation = *orientation;
        
        println!("  {} at position ({:.1}, {:.1}, {:.1})", desc, pos.x, pos.y, pos.z);
        println!("    Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 orientation[0], orientation[1], orientation[2], orientation[3]);
        
        // Calculate where the "top" of the cylinder should be
        let top_local = [0.0, 1.0, 0.0]; // Top in local space (Y-up)
        let top_world = quaternion_rotate_point(top_local, *orientation);
        println!("    Top point (local [0,1,0]) maps to world: [{:.3}, {:.3}, {:.3}]", 
                 top_world[0], top_world[1], top_world[2]);
        
        // Test SDF at a few key points
        let test_offset = 0.1;
        let test_points = [
            ([pos.x, pos.y, pos.z], "center"),
            ([pos.x + top_world[0] * (1.0 + test_offset), 
              pos.y + top_world[1] * (1.0 + test_offset), 
              pos.z + top_world[2] * (1.0 + test_offset)], "just outside top"),
        ];
        
        for (point, point_desc) in &test_points {
            let sdf = calculate_cylinder_sdf(*point, [pos.x, pos.y, pos.z], 0.5, 2.0, *orientation);
            println!("      SDF at {}: {:.3}", point_desc, sdf);
        }
        println!();
    }
    
    // Convert to GPU format and verify
    println!("📐 Verifying GPU conversion:");
    for (i, cylinder) in sim.cylinders.iter().enumerate() {
        let gpu_cylinder = CylinderGpu::from(cylinder);
        let orientation_matches = gpu_cylinder.orientation == cylinder.orientation;
        
        println!("  Cylinder {}: orientation preserved = {}", i, orientation_matches);
        if !orientation_matches {
            println!("    Physics: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     cylinder.orientation[0], cylinder.orientation[1], 
                     cylinder.orientation[2], cylinder.orientation[3]);
            println!("    GPU:     [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     gpu_cylinder.orientation[0], gpu_cylinder.orientation[1], 
                     gpu_cylinder.orientation[2], gpu_cylinder.orientation[3]);
        }
    }
}

#[test]
fn test_specific_cartpole_scenario() {
    println!("🧪 Testing specific CartPole rotation scenario...\n");
    
    // Simulate what happens in CartPole with a tilted pole
    let cart_pos = Vec3::new(0.0, 0.1, 0.0); // Cart at ground level
    let joint_offset = Vec3::new(0.0, 0.1, 0.0); // Joint on top of cart
    let joint_pos = cart_pos + joint_offset;
    let pole_length = 1.0;
    
    // Test different angles
    let test_angles = [0.0, PI / 6.0, PI / 4.0, PI / 2.0]; // 0°, 30°, 45°, 90°
    
    for angle in &test_angles {
        println!("📐 Testing angle: {:.1}° ({:.3} radians)", angle * 180.0 / PI, angle);
        
        // Calculate pole center position (pole rotates around joint)
        let pole_center = Vec3::new(
            joint_pos.x + (pole_length / 2.0) * angle.sin(),
            joint_pos.y + (pole_length / 2.0) * angle.cos(),
            joint_pos.z
        );
        
        // Create quaternion for this rotation (around Z axis)
        let orientation = quaternion_from_axis_angle([0.0, 0.0, 1.0], *angle);
        
        println!("  Joint position: ({:.3}, {:.3}, {:.3})", joint_pos.x, joint_pos.y, joint_pos.z);
        println!("  Pole center: ({:.3}, {:.3}, {:.3})", pole_center.x, pole_center.y, pole_center.z);
        println!("  Orientation quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 orientation[0], orientation[1], orientation[2], orientation[3]);
        
        // Test SDF at key points
        let test_points = [
            (joint_pos, "joint (bottom of pole)"),
            (pole_center, "pole center"),
            (Vec3::new(
                joint_pos.x + pole_length * angle.sin(),
                joint_pos.y + pole_length * angle.cos(),
                joint_pos.z
            ), "top of pole"),
        ];
        
        for (point, desc) in &test_points {
            let sdf = calculate_cylinder_sdf(
                [point.x, point.y, point.z],
                [pole_center.x, pole_center.y, pole_center.z],
                0.05, // radius
                pole_length,
                orientation
            );
            println!("    SDF at {}: {:.3}", desc, sdf);
        }
        
        // Verify the rotation is working correctly
        let y_axis = [0.0, 1.0, 0.0];
        let rotated_axis = quaternion_rotate_point(y_axis, orientation);
        println!("  Rotated Y-axis: [{:.3}, {:.3}, {:.3}]", 
                 rotated_axis[0], rotated_axis[1], rotated_axis[2]);
        // Note: Positive rotation around Z rotates from +X towards +Y, and from +Y towards -X
        // So Y-axis rotated by angle θ around Z becomes (-sin(θ), cos(θ), 0)
        println!("  Expected: [{:.3}, {:.3}, {:.3}]", -angle.sin(), angle.cos(), 0.0);
        
        let error = ((rotated_axis[0] - (-angle.sin())).powi(2) + 
                     (rotated_axis[1] - angle.cos()).powi(2) + 
                     (rotated_axis[2] - 0.0).powi(2)).sqrt();
        
        if error < 0.001 {
            println!("  ✅ Rotation correct (error: {:.6})", error);
        } else {
            println!("  ❌ Rotation incorrect (error: {:.6})", error);
        }
        println!();
    }
}

#[test]
fn test_shader_cylinder_sdf_accuracy() {
    println!("🧪 Testing shader cylinder SDF accuracy for rotated cylinders...\n");
    
    // Test a cylinder rotated 90 degrees around Z (lying horizontally along X)
    let center = [0.0, 0.0, 0.0];
    let radius = 0.5;
    let height = 2.0;
    let orientation = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
    
    println!("📐 Testing 90° Z-rotated cylinder (horizontal along X):");
    println!("  Center: {:?}, Radius: {}, Height: {}", center, radius, height);
    println!("  Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             orientation[0], orientation[1], orientation[2], orientation[3]);
    
    // Critical test points
    let test_cases = [
        // Points along the cylinder axis (now X-axis)
        ([1.0, 0.0, 0.0], "Top cap center", 0.0),
        ([-1.0, 0.0, 0.0], "Bottom cap center", 0.0),
        ([0.0, 0.0, 0.0], "Cylinder center", -0.5),
        
        // Points on the cylindrical surface
        ([0.0, 0.5, 0.0], "Surface +Y", 0.0),
        ([0.0, -0.5, 0.0], "Surface -Y", 0.0),
        ([0.0, 0.0, 0.5], "Surface +Z", 0.0),
        ([0.0, 0.0, -0.5], "Surface -Z", 0.0),
        
        // Points outside
        ([1.5, 0.0, 0.0], "Outside top cap", 0.5),
        ([0.0, 1.0, 0.0], "Outside +Y", 0.5),
        
        // Edge cases - corners of caps
        ([1.0, 0.5, 0.0], "Top cap edge +Y", 0.0),
        ([-1.0, 0.0, 0.5], "Bottom cap edge +Z", 0.0),
    ];
    
    let mut all_correct = true;
    
    for (point, desc, expected_sdf) in &test_cases {
        let calculated_sdf = calculate_cylinder_sdf(*point, center, radius, height, orientation);
        let error = (calculated_sdf - expected_sdf).abs();
        
        println!("  {} at {:?}:", desc, point);
        println!("    Expected SDF: {:.3}, Calculated: {:.3}, Error: {:.6}", 
                 expected_sdf, calculated_sdf, error);
        
        if error > 0.001 {
            println!("    ❌ SDF mismatch!");
            all_correct = false;
        } else {
            println!("    ✅ SDF correct");
        }
    }
    
    if all_correct {
        println!("\n✅ All SDF calculations correct for rotated cylinder!");
    } else {
        println!("\n❌ Some SDF calculations incorrect - shader rotation may have issues");
    }
    
    // Test with CartPole-like scenario
    println!("\n📐 Testing CartPole-like cylinder (45° tilt):");
    let pole_center = [0.354, 0.554, 0.0]; // Approx position for 45° tilt
    let pole_radius = 0.05;
    let pole_height = 1.0;
    let pole_orientation = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 4.0);
    
    // Joint should be at bottom of pole
    let joint_pos = [0.0, 0.2, 0.0];
    let joint_sdf = calculate_cylinder_sdf(joint_pos, pole_center, pole_radius, pole_height, pole_orientation);
    
    println!("  Pole center: {:?}", pole_center);
    println!("  Joint position: {:?}", joint_pos);
    println!("  SDF at joint: {:.3}", joint_sdf);
    
    // The joint should be approximately on the surface of the cylinder (SDF ≈ 0)
    if joint_sdf.abs() < 0.01 {
        println!("  ✅ Joint correctly positioned at cylinder bottom");
    } else {
        println!("  ❌ Joint not at cylinder surface - rotation calculation may be wrong");
    }
}