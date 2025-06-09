use physics::{Material, PhysicsSim, Vec3};

#[test]
fn debug_simple_collision() {
    let mut sim = PhysicsSim::new();
    
    // Create two spheres moving toward each other
    let material = Material::new(0.0, 0.5); // No friction, moderate restitution
    sim.add_sphere_with_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(1.0, 0.0, 0.0), 
        1.0, 
        material
    );
    
    sim.add_sphere_with_material(
        Vec3::new(3.0, 5.0, 0.0), 
        Vec3::new(-1.0, 0.0, 0.0), 
        1.0, 
        material
    );
    
    println!("Initial state:");
    println!("Sphere 0: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3})", 
             sim.spheres[0].pos.x, sim.spheres[0].pos.y, sim.spheres[0].pos.z,
             sim.spheres[0].vel.x, sim.spheres[0].vel.y, sim.spheres[0].vel.z);
    println!("Sphere 1: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3})", 
             sim.spheres[1].pos.x, sim.spheres[1].pos.y, sim.spheres[1].pos.z,
             sim.spheres[1].vel.x, sim.spheres[1].vel.y, sim.spheres[1].vel.z);
    
    let initial_ke = 0.5 * (
        sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2) +
        sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
    );
    println!("Initial KE: {:.3}", initial_ke);
    
    // Step simulation and monitor
    for step in 0..20 {
        sim.step_cpu();
        
        let ke = 0.5 * (
            sim.spheres[0].vel.x.powi(2) + sim.spheres[0].vel.y.powi(2) + sim.spheres[0].vel.z.powi(2) +
            sim.spheres[1].vel.x.powi(2) + sim.spheres[1].vel.y.powi(2) + sim.spheres[1].vel.z.powi(2)
        );
        
        let distance = {
            let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
            let dy = sim.spheres[1].pos.y - sim.spheres[0].pos.y;
            let dz = sim.spheres[1].pos.z - sim.spheres[0].pos.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        
        if step % 5 == 0 || distance < 2.5 || ke > initial_ke * 1.1 {
            println!("Step {}: distance={:.3}, KE={:.3}", step, distance, ke);
            println!("  Sphere 0: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3})", 
                     sim.spheres[0].pos.x, sim.spheres[0].pos.y, sim.spheres[0].pos.z,
                     sim.spheres[0].vel.x, sim.spheres[0].vel.y, sim.spheres[0].vel.z);
            println!("  Sphere 1: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3})", 
                     sim.spheres[1].pos.x, sim.spheres[1].pos.y, sim.spheres[1].pos.z,
                     sim.spheres[1].vel.x, sim.spheres[1].vel.y, sim.spheres[1].vel.z);
        }
        
        // Stop if energy explodes
        if ke > initial_ke * 2.0 {
            panic!("Energy explosion detected at step {}: KE increased from {:.3} to {:.3}", 
                   step, initial_ke, ke);
        }
    }
}

#[test]
fn debug_momentum_conservation() {
    let mut sim = PhysicsSim::new();
    
    // Create spheres with different masses
    sim.add_sphere_with_mass_and_material(
        Vec3::new(0.0, 5.0, 0.0), 
        Vec3::new(2.0, 0.0, 0.0), 
        1.0, 
        1.0, // Mass 1
        Material::new(0.0, 1.0) // Perfect elastic collision
    );
    
    sim.add_sphere_with_mass_and_material(
        Vec3::new(4.0, 5.0, 0.0), 
        Vec3::new(0.0, 0.0, 0.0), 
        1.0, 
        2.0, // Mass 2
        Material::new(0.0, 1.0)
    );
    
    // Calculate initial momentum
    let initial_momentum = sim.spheres[0].mass * sim.spheres[0].vel.x + 
                          sim.spheres[1].mass * sim.spheres[1].vel.x;
    println!("Initial momentum: {:.3}", initial_momentum);
    
    // Run simulation
    for step in 0..30 {
        sim.step_cpu();
        
        let current_momentum = sim.spheres[0].mass * sim.spheres[0].vel.x + 
                              sim.spheres[1].mass * sim.spheres[1].vel.x;
        
        let distance = {
            let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
            (dx * dx).sqrt()
        };
        
        if step % 10 == 0 || distance < 3.0 {
            println!("Step {}: momentum={:.3}, distance={:.3}", 
                     step, current_momentum, distance);
        }
        
        // Check momentum conservation
        if (current_momentum - initial_momentum).abs() > 0.001 {
            println!("Momentum conservation violated at step {}: {:.3} -> {:.3}", 
                     step, initial_momentum, current_momentum);
        }
    }
}