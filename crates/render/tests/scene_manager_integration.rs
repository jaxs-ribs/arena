use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;
use render::{CylinderGpu, Renderer, RendererConfig};

#[test] 
fn test_scene_manager_data_flow() {
    println!("🧪 Testing scene manager data flow...");
    
    // Create physics simulation with CartPole
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 4.0, // 45 degrees
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Record initial orientation
    let initial_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("🔵 Initial physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_orientation[0], initial_orientation[1], 
             initial_orientation[2], initial_orientation[3]);
    
    // Run physics steps to change orientation
    for _ in 0..3 {
        sim.step_cpu();
    }
    
    let updated_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("🔵 Updated physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             updated_orientation[0], updated_orientation[1], 
             updated_orientation[2], updated_orientation[3]);
    
    // Verify physics changed
    assert_ne!(initial_orientation, updated_orientation,
               "Physics should have updated orientation");
    
    // Test CylinderGpu conversion with updated data
    let gpu_cylinder_from_physics = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
    println!("🔵 GPU from physics: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_cylinder_from_physics.orientation[0], gpu_cylinder_from_physics.orientation[1], 
             gpu_cylinder_from_physics.orientation[2], gpu_cylinder_from_physics.orientation[3]);
    
    // Verify GPU conversion reflects updated physics
    assert_eq!(updated_orientation, gpu_cylinder_from_physics.orientation,
               "GPU conversion should match updated physics");
    
    // Test scene manager batch conversion
    let cylinder_data: Vec<CylinderGpu> = sim.cylinders.iter().map(CylinderGpu::from).collect();
    println!("🔵 Batch conversion: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder_data[cartpole.pole_idx].orientation[0], cylinder_data[cartpole.pole_idx].orientation[1], 
             cylinder_data[cartpole.pole_idx].orientation[2], cylinder_data[cartpole.pole_idx].orientation[3]);
    
    assert_eq!(updated_orientation, cylinder_data[cartpole.pole_idx].orientation,
               "Batch conversion should match updated physics");
    
    println!("✅ Scene manager data flow: PASS");
}

#[test]
fn test_live_physics_simulation_conversion() {
    println!("🧪 Testing live physics simulation conversion...");
    
    // Create simulation 
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 8.0, // Small initial angle
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("🔵 Running live simulation test...");
    
    for step in 0..10 {
        // Physics step
        sim.step_cpu();
        
        // Get current physics data
        let physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        let physics_position = sim.cylinders[cartpole.pole_idx].pos;
        
        // Convert to GPU format (simulating what scene manager does)
        let gpu_cylinders: Vec<CylinderGpu> = sim.cylinders.iter().map(CylinderGpu::from).collect();
        let gpu_orientation = gpu_cylinders[cartpole.pole_idx].orientation;
        let gpu_position = gpu_cylinders[cartpole.pole_idx].pos;
        
        // Verify data consistency at this timestep
        assert_eq!(physics_orientation, gpu_orientation,
                   "GPU orientation should match physics at step {}", step);
        
        let physics_pos_array: [f32; 3] = physics_position.into();
        assert_eq!(physics_pos_array, gpu_position,
                   "GPU position should match physics at step {}", step);
        
        if step % 3 == 0 {
            println!("🔵 Step {}: physics=[{:.3}, {:.3}, {:.3}, {:.3}], gpu=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                     step, 
                     physics_orientation[0], physics_orientation[1], physics_orientation[2], physics_orientation[3],
                     gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
        }
    }
    
    println!("✅ Live physics simulation conversion: PASS");
}

#[test]
fn test_orientation_magnitude_preservation() {
    println!("🧪 Testing orientation quaternion magnitude preservation...");
    
    // Create simulation
    let mut sim = PhysicsSim::new();
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, CartPoleConfig::default());
    
    // Run simulation and check quaternion properties
    for step in 0..5 {
        sim.step_cpu();
        
        let physics_quat = sim.cylinders[cartpole.pole_idx].orientation;
        let gpu_cylinder = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
        let gpu_quat = gpu_cylinder.orientation;
        
        // Check physics quaternion is normalized
        let physics_magnitude = (physics_quat[0].powi(2) + physics_quat[1].powi(2) + 
                                physics_quat[2].powi(2) + physics_quat[3].powi(2)).sqrt();
        assert!((physics_magnitude - 1.0).abs() < 0.001, 
                "Physics quaternion should be normalized at step {}", step);
        
        // Check GPU quaternion is normalized
        let gpu_magnitude = (gpu_quat[0].powi(2) + gpu_quat[1].powi(2) + 
                            gpu_quat[2].powi(2) + gpu_quat[3].powi(2)).sqrt();
        assert!((gpu_magnitude - 1.0).abs() < 0.001, 
                "GPU quaternion should be normalized at step {}", step);
        
        // Check they're identical
        assert_eq!(physics_quat, gpu_quat,
                   "Physics and GPU quaternions should be identical at step {}", step);
        
        if step % 2 == 0 {
            println!("🔵 Step {}: magnitude={:.6} (physics), {:.6} (gpu)", 
                     step, physics_magnitude, gpu_magnitude);
        }
    }
    
    println!("✅ Orientation magnitude preservation: PASS");
}