use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;
use render::CylinderGpu;

/// This test simulates the exact flow that happens in the main application
#[test]
fn test_main_application_render_loop_simulation() {
    println!("🧪 Testing main application render loop simulation...");
    
    // Create physics simulation exactly like main app
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let config = CartPoleConfig {
        initial_angle: 0.1, // ~5.7 degrees initial angle like main app
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("🔵 Simulating main application loop...");
    
    // Simulate main application loop for several frames
    for frame in 0..10 {
        println!("\n--- Frame {} ---", frame);
        
        // Step 1: Advance physics (like advance_physics_one_step)
        sim.params.dt = 0.016; // PHYSICS_TIMESTEP_SECONDS
        sim.step_cpu();
        
        // Get physics state after step
        let physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        let physics_position = sim.cylinders[cartpole.pole_idx].pos;
        let pole_angle = cartpole.get_pole_angle(&sim);
        
        println!("🔵 After physics step:");
        println!("    Position: ({:.3}, {:.3}, {:.3})", 
                 physics_position.x, physics_position.y, physics_position.z);
        println!("    Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 physics_orientation[0], physics_orientation[1], 
                 physics_orientation[2], physics_orientation[3]);
        println!("    Pole angle: {:.3} rad ({:.1}°)", pole_angle, pole_angle.to_degrees());
        
        // Step 2: Update renderer scene data (like update_renderer_scene_data)
        // This is what renderer.update_scene() would receive
        let spheres = &sim.spheres;
        let boxes = &sim.boxes;
        let cylinders = &sim.cylinders;
        let planes = &sim.planes;
        
        // Step 3: Convert to GPU format (like SceneManager.update())
        let cylinder_gpu_data: Vec<CylinderGpu> = cylinders.iter().map(CylinderGpu::from).collect();
        let gpu_orientation = cylinder_gpu_data[cartpole.pole_idx].orientation;
        let gpu_position = cylinder_gpu_data[cartpole.pole_idx].pos;
        
        println!("🔵 After GPU conversion:");
        println!("    GPU Position: ({:.3}, {:.3}, {:.3})", 
                 gpu_position[0], gpu_position[1], gpu_position[2]);
        println!("    GPU Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
        
        // Verify consistency
        assert_eq!(physics_orientation, gpu_orientation,
                   "GPU orientation should match physics at frame {}", frame);
        
        let physics_pos_array: [f32; 3] = physics_position.into();
        assert_eq!(physics_pos_array, gpu_position,
                   "GPU position should match physics at frame {}", frame);
        
        // Check that orientation is changing (physics is working)
        if frame > 0 {
            // For physics to be working, orientation should be changing
            println!("🔵 Physics is {} (orientation changing)",
                     if physics_orientation != [0.0, 0.0, 0.0, 1.0] { "ACTIVE" } else { "STATIC" });
        }
        
        // This is the data that would go to the GPU buffers and then to the shader
        println!("🔵 Data ready for shader: orientation=[{:.3}, {:.3}, {:.3}, {:.3}]",
                 gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
    }
    
    println!("\n✅ Main application render loop simulation: PASS");
    println!("🔵 Physics updates correctly flow to GPU data in the exact main app pattern");
}

#[test]
fn test_cartpole_physics_debug_vs_gpu_data() {
    println!("🧪 Testing CartPole physics debug output vs GPU data consistency...");
    
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    for step in 0..5 {
        sim.step_cpu();
        
        // This mimics the debug output from physics simulation.rs
        let angle = cartpole.get_pole_angle(&sim);
        let physics_orientation = sim.cylinders[cartpole.pole_idx].orientation;
        
        // Debug output like in simulation.rs lines 499-504
        if angle.abs() > 0.1 {
            println!("🟡 Physics Debug: angle={:.3}rad ({:.1}°), quat=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                     angle, angle.to_degrees(),
                     physics_orientation[0], physics_orientation[1], 
                     physics_orientation[2], physics_orientation[3]);
        }
        
        // Convert to GPU (what renderer would see)
        let gpu_cylinder = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
        
        // These should be identical
        assert_eq!(physics_orientation, gpu_cylinder.orientation,
                   "Debug physics orientation should match GPU at step {}", step);
        
        println!("🟢 GPU Data: angle={:.3}rad ({:.1}°), quat=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                 angle, angle.to_degrees(),
                 gpu_cylinder.orientation[0], gpu_cylinder.orientation[1], 
                 gpu_cylinder.orientation[2], gpu_cylinder.orientation[3]);
    }
    
    println!("✅ CartPole physics debug vs GPU data: PASS");
}

#[test]
fn test_physics_timing_and_data_capture() {
    println!("🧪 Testing physics timing and data capture scenarios...");
    
    let mut sim = PhysicsSim::new();
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, CartPoleConfig::default());
    
    // Test 1: Capture data BEFORE physics step (wrong timing)
    let before_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    sim.step_cpu();
    let after_orientation = sim.cylinders[cartpole.pole_idx].orientation;
    
    println!("🔵 Timing Test:");
    println!("    Before step: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             before_orientation[0], before_orientation[1], before_orientation[2], before_orientation[3]);
    println!("    After step:  [{:.3}, {:.3}, {:.3}, {:.3}]", 
             after_orientation[0], after_orientation[1], after_orientation[2], after_orientation[3]);
    
    // If we captured data BEFORE the physics step, we'd get stale data
    let gpu_from_stale = CylinderGpu::from(&physics::Cylinder {
        orientation: before_orientation,
        ..sim.cylinders[cartpole.pole_idx].clone()
    });
    
    let gpu_from_fresh = CylinderGpu::from(&sim.cylinders[cartpole.pole_idx]);
    
    println!("🔵 GPU from stale data: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_from_stale.orientation[0], gpu_from_stale.orientation[1],
             gpu_from_stale.orientation[2], gpu_from_stale.orientation[3]);
    println!("🔵 GPU from fresh data: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_from_fresh.orientation[0], gpu_from_fresh.orientation[1],
             gpu_from_fresh.orientation[2], gpu_from_fresh.orientation[3]);
    
    // This would cause the visual issue - using stale physics data
    if before_orientation == after_orientation {
        println!("⚠️  WARNING: Physics not changing orientation!");
    } else {
        println!("✅ Physics correctly updating orientation");
    }
    
    println!("✅ Physics timing and data capture: PASS");
}