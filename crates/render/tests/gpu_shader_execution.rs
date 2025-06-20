use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;
use render::CylinderGpu;

#[test]
fn test_gpu_shader_reads_live_orientation_data() {
    println!("🧪 Testing if GPU shader reads live orientation data...");
    
    // Create physics sim with CartPole
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    // Step physics to get changing orientation
    for i in 0..5 {
        sim.step_cpu();
        let orientation = sim.cylinders[cartpole.pole_idx].orientation;
        println!("🔵 Step {}: physics orientation=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                 i+1, orientation[0], orientation[1], orientation[2], orientation[3]);
    }
    
    // Convert to GPU format
    let cylinder_gpu = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
    println!("🔵 GPU conversion: orientation=[{:.3}, {:.3}, {:.3}, {:.3}]", 
             cylinder_gpu.orientation[0], cylinder_gpu.orientation[1], 
             cylinder_gpu.orientation[2], cylinder_gpu.orientation[3]);
    
    // Test if GPU data matches physics data
    let physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    if cylinder_gpu.orientation == physics_orientation {
        println!("✅ GPU conversion preserves orientation data");
    } else {
        println!("❌ GPU conversion loses orientation data!");
        println!("    Physics: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 physics_orientation[0], physics_orientation[1], 
                 physics_orientation[2], physics_orientation[3]);
        println!("    GPU:     [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 cylinder_gpu.orientation[0], cylinder_gpu.orientation[1], 
                 cylinder_gpu.orientation[2], cylinder_gpu.orientation[3]);
    }
}

#[test]
fn test_renderer_buffer_upload_timing() {
    println!("🧪 Testing renderer buffer upload timing...");
    
    // This test verifies the buffer upload process
    // Since we can't easily test the actual GPU execution without a full render context,
    // we'll focus on the data flow verification
    
    let mut sim = PhysicsSim::new();
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, CartPoleConfig::default());
    
    // Initial state
    let initial_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("🔵 Initial orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_orientation[0], initial_orientation[1], 
             initial_orientation[2], initial_orientation[3]);
    
    // Step and check for changes
    sim.step_cpu();
    let updated_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    println!("🔵 After step: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             updated_orientation[0], updated_orientation[1], 
             updated_orientation[2], updated_orientation[3]);
    
    if initial_orientation != updated_orientation {
        println!("✅ Physics updates orientation correctly");
        
        // Convert to GPU format
        let gpu_data = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
        if gpu_data.orientation == updated_orientation {
            println!("✅ GPU conversion happens correctly");
            println!("🔥 Issue must be in shader execution or buffer binding!");
        } else {
            println!("❌ GPU conversion fails");
        }
    } else {
        println!("❌ Physics not updating orientation");
    }
}

#[test] 
fn test_isolated_orientation_behavior() {
    println!("🧪 Testing isolated orientation behavior...");
    
    // Create cylinder and manually set orientation
    let mut sim = PhysicsSim::new();
    let cylinder_idx = sim.add_cylinder(Vec3::new(0.0, 1.0, 0.0), 0.05, 1.0, Vec3::ZERO);
    
    // Manually set a specific orientation (45 degrees around Z)
    let test_angle = std::f32::consts::PI / 4.0; // 45 degrees
    let half_angle = test_angle * 0.5;
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();
    
    sim.cylinders[cylinder_idx].orientation = [0.0, 0.0, sin_half, cos_half];
    
    println!("🔵 Set orientation to 45°: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             0.0, 0.0, sin_half, cos_half);
    
    // Convert to GPU
    let gpu_cylinder = CylinderGpu::from(&sim.cylinders[cylinder_idx]);
    println!("🔵 GPU data: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_cylinder.orientation[0], gpu_cylinder.orientation[1], 
             gpu_cylinder.orientation[2], gpu_cylinder.orientation[3]);
    
    // Verify exact match
    if gpu_cylinder.orientation == [0.0, 0.0, sin_half, cos_half] {
        println!("✅ Static orientation test passes");
    } else {
        println!("❌ Static orientation test fails");
    }
    
    // Now test that physics updates work
    sim.step_cpu();
    let after_step = sim.cylinders[cylinder_idx].orientation;
    println!("🔵 After physics step: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             after_step[0], after_step[1], after_step[2], after_step[3]);
}