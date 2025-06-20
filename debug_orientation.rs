use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::{Vec3, BodyType};

fn main() {
    println!("Testing cylinder orientation updates...");
    
    // Create simulation
    let mut sim = PhysicsSim::new();
    
    // Create CartPole
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("Initial cylinder orientation: {:?}", sim.cylinders[cartpole.pole_idx].orientation);
    
    // Run a few physics steps
    for step in 0..20 {
        sim.step_cpu();
        
        let angle = cartpole.get_pole_angle(&sim);
        let orientation = sim.cylinders[cartpole.pole_idx].orientation;
        
        if step % 5 == 0 {
            println!("Step {}: angle={:.3}rad ({:.1}°), orientation=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                     step, angle, angle.to_degrees(),
                     orientation[0], orientation[1], orientation[2], orientation[3]);
        }
    }
}