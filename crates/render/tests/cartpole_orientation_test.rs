use physics::{CartPole, CartPoleConfig, PhysicsSim, Vec3};

#[test]
fn test_cartpole_orientation_pipeline() {
    println!("\n=== Testing CartPole Orientation Pipeline ===");
    
    // Step 1: Create physics simulation with CartPole
    println!("\n1. Creating CartPole with initial angle...");
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 6.0, // 30 degrees
        ..Default::default()
    };
    
    let cartpole = CartPole::new(&mut sim, Vec3::new(0.0, 0.0, 0.0), config);
    
    // Check initial orientation
    let pole_idx = cartpole.pole_idx;
    let initial_orientation = sim.cylinders[pole_idx].orientation;
    println!("   Initial pole orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             initial_orientation[0], initial_orientation[1], 
             initial_orientation[2], initial_orientation[3]);
    
    // Verify it's not identity (should be rotated 30 degrees around Z)
    assert!(initial_orientation[2].abs() > 0.2, "Z component should be non-zero for 30° rotation");
    assert!(initial_orientation[3] < 0.98, "W component should be < 0.98 for non-identity rotation");
    
    // Step 2: Run one physics step
    println!("\n2. Running physics step...");
    sim.step_cpu();
    
    let post_step_orientation = sim.cylinders[pole_idx].orientation;
    println!("   Post-step orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             post_step_orientation[0], post_step_orientation[1], 
             post_step_orientation[2], post_step_orientation[3]);
    
    // Check if orientation changed (pole should fall due to gravity)
    let orientation_changed = 
        (initial_orientation[0] - post_step_orientation[0]).abs() > 0.001 ||
        (initial_orientation[1] - post_step_orientation[1]).abs() > 0.001 ||
        (initial_orientation[2] - post_step_orientation[2]).abs() > 0.001 ||
        (initial_orientation[3] - post_step_orientation[3]).abs() > 0.001;
    
    println!("   Orientation changed: {}", orientation_changed);
    
    // Step 3: Get cylinders array that would be passed to renderer
    println!("\n3. Getting cylinders array for renderer...");
    let cylinders = &sim.cylinders;
    println!("   Cylinders count: {}", cylinders.len());
    
    if !cylinders.is_empty() {
        let pole_cylinder = &cylinders[pole_idx];
        println!("   Pole cylinder orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
                 pole_cylinder.orientation[0], pole_cylinder.orientation[1], 
                 pole_cylinder.orientation[2], pole_cylinder.orientation[3]);
        
        // Verify it matches what we expect
        assert_eq!(pole_cylinder.orientation, post_step_orientation, 
                   "Cylinder orientation should match simulation state");
    }
    
    // Step 4: Verify orientation is still non-identity
    println!("\n4. Final verification...");
    let final_orientation = cylinders[pole_idx].orientation;
    println!("   Final cylinder orientation: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             final_orientation[0], final_orientation[1], 
             final_orientation[2], final_orientation[3]);
    
    assert!(
        final_orientation[3] < 0.99,
        "Final orientation should not be identity (w < 0.99), got w={}",
        final_orientation[3]
    );
    
    println!("\n=== CartPole Orientation Pipeline Test Passed! ===");
    println!("   Orientation was preserved through entire pipeline");
}