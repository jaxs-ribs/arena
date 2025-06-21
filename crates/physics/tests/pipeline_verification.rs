use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;

#[test]
fn test_cpu_vs_gpu_pipeline_orientation() {
    println!("🧪 Testing CPU vs GPU pipeline orientation behavior...");
    
    // Create identical simulations
    let mut cpu_sim = PhysicsSim::new();
    let mut gpu_sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 4.0, // 45 degrees
        ..Default::default()
    };
    
    let cpu_cartpole = CartPole::new(&mut cpu_sim, Vec3::ZERO, config.clone());
    let gpu_cartpole = CartPole::new(&mut gpu_sim, Vec3::ZERO, config);
    
    println!("🔵 Initial states:");
    println!("    CPU: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cpu_sim.cylinders[cpu_cartpole.pole_idx].orientation[0],
             cpu_sim.cylinders[cpu_cartpole.pole_idx].orientation[1],
             cpu_sim.cylinders[cpu_cartpole.pole_idx].orientation[2],
             cpu_sim.cylinders[cpu_cartpole.pole_idx].orientation[3]);
    println!("    GPU: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             gpu_sim.cylinders[gpu_cartpole.pole_idx].orientation[0],
             gpu_sim.cylinders[gpu_cartpole.pole_idx].orientation[1],
             gpu_sim.cylinders[gpu_cartpole.pole_idx].orientation[2],
             gpu_sim.cylinders[gpu_cartpole.pole_idx].orientation[3]);
    
    // Run CPU pipeline
    for _ in 0..3 {
        cpu_sim.step_cpu();
    }
    let cpu_orientation = cpu_sim.cylinders[cpu_cartpole.pole_idx].orientation;
    
    // Run GPU pipeline  
    gpu_sim.params.dt = 0.016;
    if let Ok(_) = gpu_sim.run(0.016, 3) {
        let gpu_orientation = gpu_sim.cylinders[gpu_cartpole.pole_idx].orientation;
        
        println!("🔵 After 3 steps:");
        println!("    CPU: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 cpu_orientation[0], cpu_orientation[1], cpu_orientation[2], cpu_orientation[3]);
        println!("    GPU: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
        
        if cpu_orientation == gpu_orientation {
            println!("✅ CPU and GPU pipelines produce SAME orientation");
        } else {
            println!("❌ CPU and GPU pipelines produce DIFFERENT orientations!");
            println!("🔥 Found the issue! GPU pipeline doesn't have orientation fixes");
        }
    } else {
        println!("⚠️ GPU pipeline failed (expected - it's for spheres, CartPole needs CPU)");
        println!("🔥 Main app must be using CPU but something else is wrong");
    }
    
    // Verify CPU has our fixes
    let initial_quat = [0.0, 0.0, 0.0, 1.0];
    if cpu_orientation != initial_quat {
        println!("✅ CPU pipeline updates orientation (our fixes work)");
    } else {
        println!("❌ CPU pipeline NOT updating orientation (our fixes not working)");
    }
}

#[test] 
fn test_main_app_pipeline_detection() {
    println!("🧪 Testing which pipeline main app should use...");
    
    // Simulate main app setup
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, CartPoleConfig::default());
    
    // Test if GPU pipeline works with CartPole
    println!("🔵 Testing GPU compatibility...");
    sim.params.dt = 0.016;
    
    match sim.run(0.016, 1) {
        Ok(_) => {
            println!("✅ GPU pipeline works with CartPole");
            let gpu_orientation = sim.cylinders[cartpole.pole_idx].orientation;
            println!("    GPU orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                     gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
        }
        Err(e) => {
            println!("❌ GPU pipeline fails with CartPole: {:?}", e);
            println!("🔥 Main app MUST use CPU pipeline");
        }
    }
    
    // Reset and test CPU pipeline
    let mut sim2 = PhysicsSim::new();
    sim2.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    let cartpole2 = CartPole::new(&mut sim2, Vec3::ZERO, CartPoleConfig::default());
    
    println!("🔵 Testing CPU pipeline...");
    sim2.step_cpu();
    let cpu_orientation = sim2.cylinders[cartpole2.pole_idx].orientation;
    println!("    CPU orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             cpu_orientation[0], cpu_orientation[1], cpu_orientation[2], cpu_orientation[3]);
}