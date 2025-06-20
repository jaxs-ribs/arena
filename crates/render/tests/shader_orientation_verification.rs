use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;

#[test]
fn test_shader_orientation_values() {
    println!("🧪 Testing shader orientation values...");
    
    // Create physics simulation with larger initial angle
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 2.0, // 90 degrees - very obvious
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("🔵 Initial 90° orientation:");
    let initial_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("    Quaternion: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_orientation[0], initial_orientation[1], initial_orientation[2], initial_orientation[3]);
    
    // Calculate expected quaternion for 90 degrees around Z-axis
    let expected_angle = std::f32::consts::PI / 2.0;
    let expected_half_angle = expected_angle * 0.5;
    let expected_sin = expected_half_angle.sin();
    let expected_cos = expected_half_angle.cos();
    
    println!("🔵 Expected 90° quaternion: [0.000, 0.000, {:.3}, {:.3}]", expected_sin, expected_cos);
    
    // Test small angle changes
    for i in 1..=5 {
        sim.step_cpu();
        let orientation = sim.cylinders[cartpole.pole_idx].orientation;
        
        // Calculate angle from quaternion
        let angle = 2.0 * orientation[2].atan2(orientation[3]);
        let degrees = angle.to_degrees();
        
        println!("🔵 Step {}: quat=[{:.3}, {:.3}, {:.3}, {:.3}] angle={:.1}°", 
                 i, orientation[0], orientation[1], orientation[2], orientation[3], degrees);
    }
    
    // Test if orientation magnitude is correct (should be 1.0 for unit quaternion)
    let final_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    let magnitude = (final_orientation[0].powi(2) + final_orientation[1].powi(2) + 
                    final_orientation[2].powi(2) + final_orientation[3].powi(2)).sqrt();
    
    println!("🔵 Quaternion magnitude: {:.6} (should be 1.0)", magnitude);
    
    if (magnitude - 1.0).abs() > 0.001 {
        println!("❌ Quaternion not normalized! This could cause shader issues.");
    } else {
        println!("✅ Quaternion properly normalized");
    }
}

#[test]
fn test_extreme_orientation_values() {
    println!("🧪 Testing extreme orientation values for shader visibility...");
    
    // Test multiple extreme angles to see which ones are most visible
    let test_angles = [
        ("30°", std::f32::consts::PI / 6.0),
        ("45°", std::f32::consts::PI / 4.0), 
        ("90°", std::f32::consts::PI / 2.0),
        ("135°", 3.0 * std::f32::consts::PI / 4.0),
    ];
    
    for (name, angle) in test_angles {
        let mut sim = PhysicsSim::new();
        let config = CartPoleConfig {
            initial_angle: angle,
            ..Default::default()
        };
        let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
        
        let orientation = sim.cylinders[cartpole.pole_idx].orientation;
        let calculated_angle = 2.0 * orientation[2].atan2(orientation[3]);
        
        println!("🔵 {} test: quat=[{:.3}, {:.3}, {:.3}, {:.3}] verified_angle={:.1}°", 
                 name, orientation[0], orientation[1], orientation[2], orientation[3], 
                 calculated_angle.to_degrees());
    }
}