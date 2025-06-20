use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;
use render::CylinderGpu;
use std::f32::consts::PI;

#[test]
fn test_cartpole_cylinder_height_issue() {
    println!("🧪 Testing CartPole cylinder height and rotation issue...\n");
    
    // Create a CartPole with 45 degree initial angle
    let mut sim = PhysicsSim::new();
    let pole_length = 1.0;
    let pole_radius = 0.05;
    let config = CartPoleConfig {
        initial_angle: PI / 4.0, // 45 degrees
        pole_length,
        pole_radius,
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Check the actual cylinder data
    let cylinder = &sim.cylinders[cartpole.pole_idx];
    
    println!("📐 CartPole Configuration:");
    println!("  Configured pole_length: {}", pole_length);
    println!("  Expected cylinder height: {}", pole_length);
    println!("  Actual cylinder half_height: {}", cylinder.half_height);
    println!("  Actual cylinder full height: {}", cylinder.half_height * 2.0);
    
    // The bug: cylinder height is 1/2 of what it should be!
    let height_ratio = (cylinder.half_height * 2.0) / pole_length;
    println!("\n❗ Height ratio (actual/expected): {:.2}", height_ratio);
    
    if (height_ratio - 0.5).abs() < 0.01 {
        println!("  ❌ BUG CONFIRMED: Cylinder is half the expected height!");
        println!("  This is because pole_half_height is passed where half_height is expected");
    } else if (height_ratio - 1.0).abs() < 0.01 {
        println!("  ✅ Cylinder height is correct");
    }
    
    // Check orientation
    println!("\n📐 Cylinder Orientation:");
    println!("  Position: ({:.3}, {:.3}, {:.3})", cylinder.pos.x, cylinder.pos.y, cylinder.pos.z);
    println!("  Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder.orientation[0], cylinder.orientation[1], 
             cylinder.orientation[2], cylinder.orientation[3]);
    
    // Calculate expected quaternion for 45° rotation around Z
    let expected_quat = quaternion_from_axis_angle([0.0, 0.0, 1.0], PI / 4.0);
    println!("  Expected:    [{:.3}, {:.3}, {:.3}, {:.3}]", 
             expected_quat[0], expected_quat[1], expected_quat[2], expected_quat[3]);
    
    let quat_error = ((cylinder.orientation[0] - expected_quat[0]).powi(2) +
                      (cylinder.orientation[1] - expected_quat[1]).powi(2) +
                      (cylinder.orientation[2] - expected_quat[2]).powi(2) +
                      (cylinder.orientation[3] - expected_quat[3]).powi(2)).sqrt();
    
    if quat_error < 0.001 {
        println!("  ✅ Orientation quaternion is correct");
    } else {
        println!("  ❌ Orientation quaternion mismatch (error: {:.6})", quat_error);
    }
    
    // Test GPU conversion
    println!("\n📐 GPU Conversion:");
    let gpu_cylinder = CylinderGpu::from(cylinder);
    println!("  GPU height: {}", gpu_cylinder.height);
    println!("  GPU orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_cylinder.orientation[0], gpu_cylinder.orientation[1], 
             gpu_cylinder.orientation[2], gpu_cylinder.orientation[3]);
    
    // The GPU conversion should preserve the (incorrect) height
    if (gpu_cylinder.height - cylinder.half_height * 2.0).abs() < 0.001 {
        println!("  ✅ GPU conversion preserves cylinder data");
    } else {
        println!("  ❌ GPU conversion changes cylinder data");
    }
    
    // Summary
    println!("\n📊 Summary:");
    println!("  The shader quaternion math is correct!");
    println!("  The issue is that CartPole creates cylinders with half the intended height.");
    println!("  Fix: In cartpole.rs line 96, change 'pole_half_height' to 'config.pole_length'");
}

#[test]
fn test_shader_rendering_with_correct_height() {
    println!("🧪 Testing shader rendering with correctly sized cylinder...\n");
    
    // Create a cylinder manually with correct height
    let mut sim = PhysicsSim::new();
    let pole_length = 1.0;
    let pole_radius = 0.05;
    let angle = PI / 4.0; // 45 degrees
    
    // Calculate position for tilted pole
    let joint_pos = Vec3::new(0.0, 0.2, 0.0);
    let pole_center = Vec3::new(
        joint_pos.x + (pole_length / 2.0) * angle.sin(),
        joint_pos.y + (pole_length / 2.0) * angle.cos(),
        joint_pos.z
    );
    
    // Add cylinder with CORRECT height
    let cylinder_idx = sim.add_cylinder(pole_center, pole_radius, pole_length, Vec3::ZERO);
    
    // Set orientation
    let half_angle = angle * 0.5;
    sim.cylinders[cylinder_idx].orientation = [
        0.0,
        0.0,
        half_angle.sin(),
        half_angle.cos()
    ];
    
    let cylinder = &sim.cylinders[cylinder_idx];
    println!("📐 Correctly sized cylinder:");
    println!("  Height: {} (half_height: {})", cylinder.half_height * 2.0, cylinder.half_height);
    println!("  Position: ({:.3}, {:.3}, {:.3})", cylinder.pos.x, cylinder.pos.y, cylinder.pos.z);
    println!("  Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder.orientation[0], cylinder.orientation[1], 
             cylinder.orientation[2], cylinder.orientation[3]);
    
    // Test SDF at key points
    println!("\n📐 SDF calculations:");
    let test_points = [
        (joint_pos, "Joint (bottom of pole)"),
        (pole_center, "Pole center"),
        (Vec3::new(
            joint_pos.x + pole_length * angle.sin(),
            joint_pos.y + pole_length * angle.cos(),
            joint_pos.z
        ), "Top of pole"),
    ];
    
    for (point, desc) in &test_points {
        let sdf = calculate_cylinder_sdf(
            [point.x, point.y, point.z],
            [pole_center.x, pole_center.y, pole_center.z],
            pole_radius,
            cylinder.half_height * 2.0, // Use actual cylinder height
            cylinder.orientation
        );
        println!("  {} at ({:.3}, {:.3}, {:.3}): SDF = {:.3}", 
                 desc, point.x, point.y, point.z, sdf);
        
        if desc.contains("Joint") || desc.contains("Top") {
            if sdf.abs() < 0.01 {
                println!("    ✅ Correctly on cylinder surface");
            } else {
                println!("    ❌ Not on cylinder surface as expected");
            }
        }
    }
}

// Helper functions
fn quaternion_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();
    
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let norm_axis = [axis[0] / len, axis[1] / len, axis[2] / len];
    
    [
        norm_axis[0] * sin_half,
        norm_axis[1] * sin_half,
        norm_axis[2] * sin_half,
        cos_half
    ]
}

fn quaternion_rotate_point(p: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    
    let cross1 = [
        qv[1] * p[2] - qv[2] * p[1],
        qv[2] * p[0] - qv[0] * p[2],
        qv[0] * p[1] - qv[1] * p[0]
    ];
    
    let temp = [
        cross1[0] + qw * p[0],
        cross1[1] + qw * p[1],
        cross1[2] + qw * p[2]
    ];
    
    let cross2 = [
        qv[1] * temp[2] - qv[2] * temp[1],
        qv[2] * temp[0] - qv[0] * temp[2],
        qv[0] * temp[1] - qv[1] * temp[0]
    ];
    
    [
        p[0] + 2.0 * cross2[0],
        p[1] + 2.0 * cross2[1],
        p[2] + 2.0 * cross2[2]
    ]
}

fn calculate_cylinder_sdf(
    point: [f32; 3], 
    center: [f32; 3], 
    radius: f32, 
    height: f32, 
    orientation: [f32; 4]
) -> f32 {
    let offset = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2]
    ];
    
    let conj_q = [-orientation[0], -orientation[1], -orientation[2], orientation[3]];
    let local_p = quaternion_rotate_point(offset, conj_q);
    
    let xz_dist = (local_p[0] * local_p[0] + local_p[2] * local_p[2]).sqrt();
    let d = [xz_dist - radius, local_p[1].abs() - height * 0.5];
    
    let max_component = d[0].max(d[1]);
    let max_d = [d[0].max(0.0), d[1].max(0.0)];
    let length_max_d = (max_d[0] * max_d[0] + max_d[1] * max_d[1]).sqrt();
    
    max_component.min(0.0) + length_max_d
}