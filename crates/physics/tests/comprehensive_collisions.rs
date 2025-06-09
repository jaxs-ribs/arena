use physics::{Material, PhysicsSim, Vec3};

/// Test momentum conservation in elastic collisions
#[test]
fn test_momentum_conservation() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0); // No gravity to isolate collision effects
    
    // Perfect elastic collision
    let elastic_material = Material::new(0.0, 1.0);
    
    // Moving sphere hitting stationary sphere
    sim.add_sphere_with_material(
        Vec3::new(0.0, 0.0, 0.0), 
        Vec3::new(2.0, 0.0, 0.0), 
        1.0, 
        elastic_material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(3.0, 0.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        elastic_material
    );
    
    let initial_momentum = sim.spheres[0].mass * sim.spheres[0].vel.x + 
                          sim.spheres[1].mass * sim.spheres[1].vel.x;
    
    // Run until collision occurs and resolves
    for _ in 0..50 {
        sim.step_cpu();
    }
    
    let final_momentum = sim.spheres[0].mass * sim.spheres[0].vel.x + 
                        sim.spheres[1].mass * sim.spheres[1].vel.x;
    
    // Momentum should be conserved (within numerical precision)
    assert!((final_momentum - initial_momentum).abs() < 0.01, 
            "Momentum not conserved: {:.3} -> {:.3}", initial_momentum, final_momentum);
}

/// Test energy conservation in perfectly elastic collision
#[test]
fn test_energy_conservation_elastic() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0); // No gravity
    
    let elastic_material = Material::new(0.0, 1.0); // Perfect elastic
    
    sim.add_sphere_with_material(
        Vec3::new(0.0, 0.0, 0.0), 
        Vec3::new(1.0, 0.0, 0.0), 
        1.0, 
        elastic_material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(2.5, 0.0, 0.0), 
        Vec3::new(-1.0, 0.0, 0.0), 
        1.0, 
        elastic_material
    );
    
    let initial_ke = 0.5 * sim.spheres[0].mass * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2)
    ) + 0.5 * sim.spheres[1].mass * (
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    
    // Run simulation
    for _ in 0..50 {
        sim.step_cpu();
    }
    
    let final_ke = 0.5 * sim.spheres[0].mass * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2)
    ) + 0.5 * sim.spheres[1].mass * (
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    
    // Energy should be conserved in elastic collision
    assert!((final_ke - initial_ke).abs() < 0.01, 
            "Energy not conserved in elastic collision: {:.3} -> {:.3}", initial_ke, final_ke);
}

/// Test energy dissipation in inelastic collision
#[test]
fn test_energy_dissipation_inelastic() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0); // No gravity
    
    let inelastic_material = Material::new(0.0, 0.0); // Perfectly inelastic
    
    sim.add_sphere_with_material(
        Vec3::new(0.0, 0.0, 0.0), 
        Vec3::new(1.0, 0.0, 0.0), 
        1.0, 
        inelastic_material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(2.5, 0.0, 0.0), 
        Vec3::new(-1.0, 0.0, 0.0), 
        1.0, 
        inelastic_material
    );
    
    let initial_ke = 0.5 * sim.spheres[0].mass * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2)
    ) + 0.5 * sim.spheres[1].mass * (
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    
    // Run simulation
    for _ in 0..50 {
        sim.step_cpu();
    }
    
    let final_ke = 0.5 * sim.spheres[0].mass * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2)
    ) + 0.5 * sim.spheres[1].mass * (
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    
    // Energy should be lost in inelastic collision
    assert!(final_ke < initial_ke, 
            "Energy should be lost in inelastic collision: {:.3} -> {:.3}", initial_ke, final_ke);
    
    // But shouldn't be completely lost
    assert!(final_ke > 0.0, "Some energy should remain after collision");
}

/// Test different mass ratios in collisions
#[test]
fn test_mass_ratio_effects() {
    // Heavy sphere hitting light sphere
    let mut sim1 = PhysicsSim::new();
    sim1.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    
    let material = Material::new(0.0, 0.8);
    
    sim1.add_sphere_with_mass_and_material(
        Vec3::new(0.0, 0.0, 0.0), 
        Vec3::new(1.0, 0.0, 0.0), 
        1.0, 
        10.0, // Heavy
        material
    );
    
    sim1.add_sphere_with_mass_and_material(
        Vec3::new(2.5, 0.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        1.0, // Light
        material
    );
    
    for _ in 0..50 {
        sim1.step_cpu();
    }
    
    // Heavy sphere should continue moving forward (though slower)
    assert!(sim1.spheres[0].vel.x > 0.0, "Heavy sphere should continue moving forward");
    
    // Light sphere should be moving fast forward
    assert!(sim1.spheres[1].vel.x > sim1.spheres[0].vel.x, 
            "Light sphere should move faster than heavy sphere after collision");
}

/// Test collision separation prevents overlap
#[test]
fn test_collision_separation() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    
    let material = Material::new(0.0, 0.5);
    
    // Place spheres very close together (overlapping)
    sim.add_sphere_with_material(
        Vec3::new(0.0, 0.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(1.5, 0.0, 0.0), // Overlapping by 0.5 units
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        material
    );
    
    // Run simulation to resolve overlap
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    // Check that spheres are properly separated
    let distance = {
        let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
        let dy = sim.spheres[1].pos.y - sim.spheres[0].pos.y;
        let dz = sim.spheres[1].pos.z - sim.spheres[0].pos.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    
    let min_distance = sim.spheres[0].radius + sim.spheres[1].radius;
    assert!(distance >= min_distance - 0.01, 
            "Spheres should be separated: distance={:.3}, min_distance={:.3}", 
            distance, min_distance);
}

/// Test multiple sphere collision handling
#[test]
fn test_multiple_sphere_collisions() {
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    
    let material = Material::new(0.0, 0.7);
    
    // Create a line of spheres
    for i in 0..5 {
        let velocity = if i == 0 { Vec3::new(2.0, 0.0, 0.0) } else { Vec3::new(0.0, 0.0, 0.0) };
        sim.add_sphere_with_material(
            Vec3::new(i as f32 * 2.1, 0.0, 0.0), 
            velocity, 
            1.0, 
            material
        );
    }
    
    // Run simulation
    for _ in 0..100 {
        sim.step_cpu();
    }
    
    // Last sphere should be moving (energy should propagate through the line)
    assert!(sim.spheres[4].vel.x > 0.1, 
            "Energy should propagate to last sphere: vel={:.3}", sim.spheres[4].vel.x);
    
    // First sphere should have slowed down significantly
    assert!(sim.spheres[0].vel.x < 1.0, 
            "First sphere should have slowed down: vel={:.3}", sim.spheres[0].vel.x);
}

/// Test spatial grid efficiency with many spheres
#[test]
fn test_spatial_grid_performance() {
    let mut sim = PhysicsSim::new();
    
    let material = Material::default();
    
    // Add many spheres in a grid pattern
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..3 {
                sim.add_sphere_with_material(
                    Vec3::new(x as f32 * 3.0, y as f32 * 3.0 + 10.0, z as f32 * 3.0),
                    Vec3::new(
                        (x as f32 - 2.0) * 0.5, 
                        0.0, 
                        (z as f32 - 1.0) * 0.5
                    ),
                    1.0,
                    material
                );
            }
        }
    }
    
    // Run simulation for several steps
    let start_time = std::time::Instant::now();
    for _ in 0..10 {
        sim.step_cpu();
    }
    let duration = start_time.elapsed();
    
    println!("Simulated {} spheres for 10 steps in {:.3}ms", 
             sim.spheres.len(), duration.as_millis());
    
    // Should complete in reasonable time (less than 1 second for 75 spheres)
    assert!(duration.as_millis() < 1000, 
            "Simulation with {} spheres took too long: {}ms", 
            sim.spheres.len(), duration.as_millis());
    
    // Check spatial grid utilization
    let (total_cells, occupied_cells, occupancy_ratio) = sim.spatial_grid_stats();
    println!("Spatial grid: {}/{} cells occupied ({:.1}%)", 
             occupied_cells, total_cells, occupancy_ratio * 100.0);
    
    assert!(occupied_cells > 0, "Spatial grid should have occupied cells");
    assert!(occupancy_ratio < 0.5, "Spatial grid should not be overly dense");
}