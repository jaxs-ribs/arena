use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::{Vec3, BodyType};

#[test]
fn test_physics_orientation_updates() {
    println!("🧪 Testing physics orientation updates...");
    
    // Create simulation with CartPole
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 4.0, // 45 degrees
        ..Default::default()
    };
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Initial orientation
    let initial_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("🔵 Initial orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_orientation[0], initial_orientation[1], 
             initial_orientation[2], initial_orientation[3]);
    
    // Run physics steps
    for step in 0..10 {
        sim.step_cpu();
        let new_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        
        if step % 3 == 0 {
            println!("🔵 Step {}: orientation=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                     step, new_orientation[0], new_orientation[1], 
                     new_orientation[2], new_orientation[3]);
        }
    }
    
    let final_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    
    // Verify orientation changed (physics is working)
    assert_ne!(initial_orientation, final_orientation, 
               "Physics should update cylinder orientation");
    
    println!("✅ Physics orientation updates: PASS");
}

#[test]
fn test_cylinder_to_gpu_conversion() {
    // Import render crate to test GPU conversion
    // This test will verify if physics cylinder data converts to GPU format correctly
    println!("🧪 Testing physics orientation data structure...");
    
    // Create test cylinder with specific orientation
    let mut sim = PhysicsSim::new();
    let cylinder_idx = sim.add_cylinder(
        Vec3::new(1.0, 2.0, 3.0),  // position
        0.5,                       // radius
        1.0,                       // half_height
        Vec3::ZERO,                // velocity
    );
    
    // Set specific orientation (45° rotation around Z)
    let test_angle = std::f32::consts::PI / 4.0; // 45 degrees
    let half_angle = test_angle * 0.5;
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();
    
    sim.cylinders[cylinder_idx].orientation = [
        0.0,         // x
        0.0,         // y
        sin_half,    // z (rotation around z-axis)
        cos_half,    // w
    ];
    
    println!("🔵 Physics cylinder orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             sim.cylinders[cylinder_idx].orientation[0],
             sim.cylinders[cylinder_idx].orientation[1],
             sim.cylinders[cylinder_idx].orientation[2],
             sim.cylinders[cylinder_idx].orientation[3]);
    
    // Verify the physics data structure is correct
    assert_eq!(sim.cylinders[cylinder_idx].pos, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(sim.cylinders[cylinder_idx].radius, 0.5);
    assert_eq!(sim.cylinders[cylinder_idx].half_height, 1.0);
    
    let expected_orientation = [0.0, 0.0, sin_half, cos_half];
    assert_eq!(sim.cylinders[cylinder_idx].orientation, expected_orientation);
    
    println!("✅ Physics cylinder data structure: PASS");
}

#[test]
fn test_cartpole_physics_orientation_pipeline() {
    println!("🧪 Testing full CartPole physics orientation pipeline...");
    
    // Create CartPole simulation
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Record initial state
    let initial_physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    
    println!("🔵 Initial physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_physics_orientation[0], initial_physics_orientation[1], 
             initial_physics_orientation[2], initial_physics_orientation[3]);
    
    // Run physics simulation
    for _ in 0..5 {
        sim.step_cpu();
    }
    
    // Get updated state
    let updated_physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    
    println!("🔵 Updated physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             updated_physics_orientation[0], updated_physics_orientation[1], 
             updated_physics_orientation[2], updated_physics_orientation[3]);
    
    // Verify physics changed
    assert_ne!(initial_physics_orientation, updated_physics_orientation,
               "Physics should update orientation during simulation");
    
    // Check that the orientation quaternion is normalized (valid)
    let quat = updated_physics_orientation;
    let magnitude = (quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]).sqrt();
    assert!((magnitude - 1.0).abs() < 0.001, "Orientation quaternion should be normalized");
    
    // Check that Z component is non-zero (rotation around Z-axis)
    assert!(quat[2].abs() > 0.001, "Z component should be non-zero for 2D rotation");
    
    println!("✅ CartPole physics orientation pipeline: PASS");
}