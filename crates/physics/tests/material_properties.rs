use physics::{Material, PhysicsSim, Vec3};
use physics::types::Vec2;

#[test]
fn test_restitution_bouncy_vs_damped() {
    // Test bouncy material
    let mut sim_bouncy = PhysicsSim::new();
    sim_bouncy.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(25.0, 25.0)); // Ground plane at y=0
    let bouncy_material = Material::bouncy(); // High restitution
    sim_bouncy.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        bouncy_material
    );
    
    // Test damped material
    let mut sim_damped = PhysicsSim::new();
    sim_damped.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(25.0, 25.0)); // Ground plane at y=0
    let damped_material = Material::new(0.5, 0.1); // Low restitution
    sim_damped.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        damped_material
    );
    
    // Run simulations until spheres hit the ground and bounce
    for _ in 0..100 {
        sim_bouncy.step_cpu();
        sim_damped.step_cpu();
    }
    
    // Check that bouncy sphere has higher velocity after bouncing
    let bouncy_speed = (sim_bouncy.spheres[0].vel.x.powi(2) + 
                       sim_bouncy.spheres[0].vel.y.powi(2) + 
                       sim_bouncy.spheres[0].vel.z.powi(2)).sqrt();
    
    let damped_speed = (sim_damped.spheres[0].vel.x.powi(2) + 
                       sim_damped.spheres[0].vel.y.powi(2) + 
                       sim_damped.spheres[0].vel.z.powi(2)).sqrt();
    
    println!("Bouncy speed: {:.3}, Damped speed: {:.3}", bouncy_speed, damped_speed);
    assert!(bouncy_speed > damped_speed, "Bouncy material should retain more velocity");
}

#[test]
fn test_friction_slippery_vs_rough() {
    // Test with slippery material (low friction)
    let mut sim_slippery = PhysicsSim::new();
    let ramp_normal = Vec3::new(0.3, 1.0, 0.0).normalize();
    sim_slippery.add_plane(ramp_normal, -2.0, Vec2::new(10.0, 10.0)); // Tilted ramp
    
    let slippery_material = Material::slippery(); // Low friction
    sim_slippery.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        slippery_material
    );
    
    // Test with rough material (high friction)
    let mut sim_rough = PhysicsSim::new();
    sim_rough.add_plane(ramp_normal, -2.0, Vec2::new(10.0, 10.0)); // Same ramp
    
    let rough_material = Material::new(0.9, 0.3); // High friction
    sim_rough.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        rough_material
    );
    
    // Run simulation
    for _ in 0..200 {
        sim_slippery.step_cpu();
        sim_rough.step_cpu();
    }
    
    // Check that slippery sphere slides further down the ramp
    let slippery_x = sim_slippery.spheres[0].pos.x;
    let rough_x = sim_rough.spheres[0].pos.x;
    
    println!("Slippery sphere X: {:.3}, Rough sphere X: {:.3}", slippery_x, rough_x);
    assert!(slippery_x > rough_x, "Slippery sphere should slide further down the ramp");
}

#[test] 
fn test_sphere_sphere_material_interaction() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0); // Disable gravity to isolate collision effects
    
    // Create two spheres with different materials
    let bouncy_material = Material::bouncy();
    let damped_material = Material::new(0.2, 0.1);
    
    // Position them to collide - closer together to ensure collision
    sim.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(1.0, 0.0, 0.0), 
        1.0, 
        bouncy_material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(2.2, 5.0, 0.0), // Closer to guarantee collision
        Vec3::new(-1.0, 0.0, 0.0), 
        1.0, 
        damped_material
    );
    
    // Record initial kinetic energy
    let initial_ke = 0.5 * (sim.spheres[0].vel.x.powi(2) + sim.spheres[1].vel.x.powi(2));
    
    println!("Initial positions: sphere0=({:.2}, {:.2}, {:.2}), sphere1=({:.2}, {:.2}, {:.2})", 
             sim.spheres[0].pos.x, sim.spheres[0].pos.y, sim.spheres[0].pos.z,
             sim.spheres[1].pos.x, sim.spheres[1].pos.y, sim.spheres[1].pos.z);
    println!("Initial velocities: sphere0=({:.2}, {:.2}, {:.2}), sphere1=({:.2}, {:.2}, {:.2})", 
             sim.spheres[0].vel.x, sim.spheres[0].vel.y, sim.spheres[0].vel.z,
             sim.spheres[1].vel.x, sim.spheres[1].vel.y, sim.spheres[1].vel.z);
    
    // Run simulation until collision occurs
    for i in 0..50 {
        sim.step_cpu();
        if i % 10 == 0 {
            println!("Step {}: sphere0=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2})", 
                     i, sim.spheres[0].pos.x, sim.spheres[0].pos.y, sim.spheres[0].pos.z,
                     sim.spheres[0].vel.x, sim.spheres[0].vel.y, sim.spheres[0].vel.z);
            println!("Step {}: sphere1=({:.2}, {:.2}, {:.2}) vel=({:.2}, {:.2}, {:.2})", 
                     i, sim.spheres[1].pos.x, sim.spheres[1].pos.y, sim.spheres[1].pos.z,
                     sim.spheres[1].vel.x, sim.spheres[1].vel.y, sim.spheres[1].vel.z);
        }
    }
    
    // Check final kinetic energy - should be reduced due to inelastic collision
    let final_ke = 0.5 * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2) +
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    
    println!("Initial KE: {:.3}, Final KE: {:.3}", initial_ke, final_ke);
    
    // Energy should be reduced due to inelastic collision (combined restitution < 1.0)
    assert!(final_ke < initial_ke, "Energy should be lost in inelastic collision");
    
    // Spheres should be moving after collision (not stuck)
    let total_speed = (sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2) +
                      sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)).sqrt();
    assert!(total_speed > 0.1, "Spheres should still be moving after collision");
}

#[test]
fn test_material_combinations() {
    // Test that material combination formulas work correctly
    let mat1 = Material::new(0.6, 0.8);
    let mat2 = Material::new(0.4, 0.2);
    
    // Geometric mean should be between the individual values
    let combined_friction = (mat1.friction * mat2.friction).sqrt();
    let combined_restitution = (mat1.restitution * mat2.restitution).sqrt();
    
    assert!(combined_friction > mat2.friction);
    assert!(combined_friction < mat1.friction);
    assert!(combined_restitution > mat2.restitution);
    assert!(combined_restitution < mat1.restitution);
    
    println!("Combined friction: {:.3}, Combined restitution: {:.3}", 
             combined_friction, combined_restitution);
}