use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;
use render::CylinderGpu;

#[test]
fn test_cylinder_gpu_conversion_preserves_orientation() {
    println!("🧪 Testing Cylinder → CylinderGpu conversion...");
    
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
    
    // Convert to GPU format
    let gpu_cylinder = CylinderGpu::from(&sim.cylinders[cylinder_idx]);
    
    println!("🔵 GPU cylinder orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_cylinder.orientation[0],
             gpu_cylinder.orientation[1],
             gpu_cylinder.orientation[2],
             gpu_cylinder.orientation[3]);
    
    // Verify conversion preserves orientation exactly
    assert_eq!(sim.cylinders[cylinder_idx].orientation, gpu_cylinder.orientation,
               "GPU conversion should preserve orientation exactly");
    
    // Verify position conversion
    let expected_pos: [f32; 3] = sim.cylinders[cylinder_idx].pos.into();
    assert_eq!(expected_pos, gpu_cylinder.pos,
               "GPU conversion should preserve position");
    
    // Verify height conversion (half_height * 2)
    let expected_height = sim.cylinders[cylinder_idx].half_height * 2.0;
    assert_eq!(expected_height, gpu_cylinder.height,
               "GPU conversion should double half_height");
    
    println!("✅ Cylinder → CylinderGpu conversion: PASS");
}

#[test]
fn test_cartpole_physics_to_gpu_conversion() {
    println!("🧪 Testing CartPole physics → GPU conversion...");
    
    // Create CartPole simulation
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Record initial state
    let initial_physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    let initial_gpu_cylinder = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
    
    println!("🔵 Initial physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_physics_orientation[0], initial_physics_orientation[1], 
             initial_physics_orientation[2], initial_physics_orientation[3]);
    println!("🔵 Initial GPU orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_gpu_cylinder.orientation[0], initial_gpu_cylinder.orientation[1], 
             initial_gpu_cylinder.orientation[2], initial_gpu_cylinder.orientation[3]);
    
    // Verify initial conversion
    assert_eq!(initial_physics_orientation, initial_gpu_cylinder.orientation,
               "Initial GPU orientation should match physics");
    
    // Run physics simulation
    for step in 0..5 {
        sim.step_cpu();
        
        if step % 2 == 0 {
            let current_physics = sim.cylinders[cartpole.pole_idx].orientation;
            let current_gpu = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
            
            println!("🔵 Step {} physics: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     step, current_physics[0], current_physics[1], current_physics[2], current_physics[3]);
            println!("🔵 Step {} GPU: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     step, current_gpu.orientation[0], current_gpu.orientation[1], 
                     current_gpu.orientation[2], current_gpu.orientation[3]);
            
            // Verify conversion is always exact
            assert_eq!(current_physics, current_gpu.orientation,
                       "GPU orientation should always match physics at step {}", step);
        }
    }
    
    // Get final state
    let final_physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    let final_gpu_cylinder = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
    
    println!("🔵 Final physics orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             final_physics_orientation[0], final_physics_orientation[1], 
             final_physics_orientation[2], final_physics_orientation[3]);
    println!("🔵 Final GPU orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             final_gpu_cylinder.orientation[0], final_gpu_cylinder.orientation[1], 
             final_gpu_cylinder.orientation[2], final_gpu_cylinder.orientation[3]);
    
    // Verify physics changed
    assert_ne!(initial_physics_orientation, final_physics_orientation,
               "Physics should update orientation during simulation");
    
    // Verify GPU reflects physics exactly
    assert_eq!(final_physics_orientation, final_gpu_cylinder.orientation,
               "Final GPU orientation should exactly match physics orientation");
    
    // Verify GPU conversion updated
    assert_ne!(initial_gpu_cylinder.orientation, final_gpu_cylinder.orientation,
               "GPU orientation should change when physics changes");
    
    println!("✅ CartPole physics → GPU conversion: PASS");
}

#[test]
fn test_multiple_cylinders_gpu_conversion() {
    println!("🧪 Testing multiple cylinders GPU conversion...");
    
    // Create simulation with multiple cylinders
    let mut sim = PhysicsSim::new();
    
    // Create cylinders with different orientations
    let cylinder1_idx = sim.add_cylinder(Vec3::new(0.0, 1.0, 0.0), 0.3, 0.8, Vec3::ZERO);
    let cylinder2_idx = sim.add_cylinder(Vec3::new(2.0, 1.0, 0.0), 0.4, 1.2, Vec3::ZERO);
    
    // Set different orientations
    sim.cylinders[cylinder1_idx].orientation = [0.0, 0.0, 0.383, 0.924]; // ~45°
    sim.cylinders[cylinder2_idx].orientation = [0.0, 0.0, 0.707, 0.707]; // ~90°
    
    // Convert all to GPU format
    let gpu_cylinders: Vec<CylinderGpu> = sim.cylinders.iter().map(CylinderGpu::from).collect();
    
    println!("🔵 Cylinder 1 - Physics: [{:.3}, {:.3}, {:.3}, {:.3}] → GPU: [{:.3}, {:.3}, {:.3}, {:.3}]",
             sim.cylinders[cylinder1_idx].orientation[0], sim.cylinders[cylinder1_idx].orientation[1],
             sim.cylinders[cylinder1_idx].orientation[2], sim.cylinders[cylinder1_idx].orientation[3],
             gpu_cylinders[cylinder1_idx].orientation[0], gpu_cylinders[cylinder1_idx].orientation[1],
             gpu_cylinders[cylinder1_idx].orientation[2], gpu_cylinders[cylinder1_idx].orientation[3]);
    
    println!("🔵 Cylinder 2 - Physics: [{:.3}, {:.3}, {:.3}, {:.3}] → GPU: [{:.3}, {:.3}, {:.3}, {:.3}]",
             sim.cylinders[cylinder2_idx].orientation[0], sim.cylinders[cylinder2_idx].orientation[1],
             sim.cylinders[cylinder2_idx].orientation[2], sim.cylinders[cylinder2_idx].orientation[3],
             gpu_cylinders[cylinder2_idx].orientation[0], gpu_cylinders[cylinder2_idx].orientation[1],
             gpu_cylinders[cylinder2_idx].orientation[2], gpu_cylinders[cylinder2_idx].orientation[3]);
    
    // Verify both conversions
    assert_eq!(sim.cylinders[cylinder1_idx].orientation, gpu_cylinders[cylinder1_idx].orientation);
    assert_eq!(sim.cylinders[cylinder2_idx].orientation, gpu_cylinders[cylinder2_idx].orientation);
    
    // Verify they're different
    assert_ne!(gpu_cylinders[cylinder1_idx].orientation, gpu_cylinders[cylinder2_idx].orientation);
    
    println!("✅ Multiple cylinders GPU conversion: PASS");
}