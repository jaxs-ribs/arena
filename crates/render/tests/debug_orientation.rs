use physics::{PhysicsSim, Vec3, CartPole, CartPoleConfig};

#[test]
fn debug_cartpole_orientation() {
    // Create simulation
    let mut sim = PhysicsSim::new();
    
    // Create CartPole with 30-degree initial angle
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0,
        ..Default::default()
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::new(0.0, 0.0, 0.0), config);
    let pole_idx = cartpole.pole_idx;
    
    // Check initial state
    let initial = sim.cylinders[pole_idx].orientation;
    eprintln!("Initial orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial[0], initial[1], initial[2], initial[3]);
    
    // The orientation should be a 30-degree rotation around Z axis
    // For a quaternion rotating θ around Z: q = [0, 0, sin(θ/2), cos(θ/2)]
    // For 30° = π/6: sin(π/12) ≈ 0.259, cos(π/12) ≈ 0.966
    
    assert!(initial[0].abs() < 0.01, "X should be ~0");
    assert!(initial[1].abs() < 0.01, "Y should be ~0");
    assert!((initial[2] - 0.259).abs() < 0.01, "Z should be ~0.259");
    assert!((initial[3] - 0.966).abs() < 0.01, "W should be ~0.966");
    
    // Run physics step
    sim.step_cpu();
    
    let after_step = sim.cylinders[pole_idx].orientation;
    eprintln!("After physics step: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             after_step[0], after_step[1], after_step[2], after_step[3]);
    
    // The orientation should have changed (pole falling)
    let changed = (initial[0] - after_step[0]).abs() > 0.0001 ||
                  (initial[1] - after_step[1]).abs() > 0.0001 ||
                  (initial[2] - after_step[2]).abs() > 0.0001 ||
                  (initial[3] - after_step[3]).abs() > 0.0001;
    
    assert!(changed, "Orientation should change during physics step");
    
    // Verify it's still non-identity
    assert!(after_step[3] < 0.999, "Should not be identity quaternion");
    
    eprintln!("✓ Orientation is preserved correctly in physics simulation");
}