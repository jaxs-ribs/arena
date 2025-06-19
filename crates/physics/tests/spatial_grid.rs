use physics::{BoundingBox, PhysicsSim, Sphere, SpatialGrid, Vec3};

#[test]
fn test_spatial_grid_basic() {
    // Create a small grid for testing
    let bounds = BoundingBox {
        min: Vec3::new(-10.0, -10.0, -10.0),
        max: Vec3::new(10.0, 10.0, 10.0),
    };
    let mut grid = SpatialGrid::new(4.0, bounds);
    
    // Create some test spheres
    let spheres = vec![
        Sphere::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0),    // Center
        Sphere::new(Vec3::new(1.5, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0),    // Close to first
        Sphere::new(Vec3::new(8.0, 8.0, 8.0), Vec3::new(0.0, 0.0, 0.0), 1.0),    // Far away
    ];
    
    // Update grid with spheres
    grid.update(&spheres);
    
    // Get potential collision pairs
    let pairs = grid.get_potential_pairs();
    
    // Should find the two close spheres as a potential pair
    assert!(!pairs.is_empty(), "Should find at least one potential collision pair");
    
    // Verify the close spheres are detected as potential colliders
    let contains_close_pair = pairs.iter().any(|(i, j)| {
        (*i == 0 && *j == 1) || (*i == 1 && *j == 0)
    });
    assert!(contains_close_pair, "Should detect the two close spheres as potential colliders");
}

#[test]
fn test_spatial_grid_performance() {
    let mut sim = PhysicsSim::new();
    
    // Add many spheres in a grid pattern
    for x in 0..10 {
        for y in 0..10 {
            for z in 0..5 {
                sim.add_sphere(
                    Vec3::new(x as f32 * 3.0, y as f32 * 3.0, z as f32 * 3.0),
                    Vec3::new(0.0, 0.0, 0.0),
                    1.0
                );
            }
        }
    }
    
    // Run a few simulation steps to test performance
    for _ in 0..10 {
        sim.step_cpu();
    }
    
    // Check spatial grid stats
    let (occupied_cells, total_entries, avg_entries_per_cell) = sim.spatial_grid_stats();
    let total_cells = sim.spatial_grid.cells.len();
    let occupancy_ratio = occupied_cells as f32 / total_cells as f32;
    
    println!("Spatial grid stats: {}/{} cells occupied ({:.2}%), avg entries per cell: {:.1}", 
             occupied_cells, total_cells, occupancy_ratio * 100.0, avg_entries_per_cell);
    
    // Verify grid is being used efficiently
    assert!(occupied_cells > 0, "Grid should have occupied cells");
    assert!(occupancy_ratio < 1.0, "Grid should not be completely full");
}

#[test]
fn test_sphere_collision_with_spatial_grid() {
    let mut sim = PhysicsSim::new();
    
    // Add two spheres that should collide
    sim.add_sphere(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0);
    sim.add_sphere(Vec3::new(1.8, 5.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0);
    
    // Record initial positions
    let initial_distance = {
        let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
        let dy = sim.spheres[1].pos.y - sim.spheres[0].pos.y;
        let dz = sim.spheres[1].pos.z - sim.spheres[0].pos.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    
    // Step simulation to resolve collision
    sim.step_cpu();
    
    // Check final distance after collision resolution
    let final_distance = {
        let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
        let dy = sim.spheres[1].pos.y - sim.spheres[0].pos.y;
        let dz = sim.spheres[1].pos.z - sim.spheres[0].pos.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    
    println!("Initial distance: {}, Final distance: {}", initial_distance, final_distance);
    
    // Spheres should be pushed apart
    // The collision resolution uses 80% position correction with 0.01 slop
    // so we need to be more lenient with our expectations
    assert!(final_distance >= 1.9, "Spheres should be mostly separated");
    assert!(final_distance > initial_distance, "Spheres should be pushed further apart");
}