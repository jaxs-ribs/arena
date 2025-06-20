//! Debug grid layout to understand the overlap issue

use physics::{PhysicsSim, CartPoleGrid, CartPoleConfig, Vec3, Vec2};

#[test]
fn debug_grid_positions() {
    let mut sim = PhysicsSim::new();
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(50.0, 50.0));
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.1,
        pole_mass: 0.1,
        initial_angle: 0.1,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    // Create 2x3 grid with 2.5 spacing
    let grid = CartPoleGrid::new(&mut sim, 2, 3, 2.5, config);
    
    println!("\nGrid layout debug:");
    println!("Rows: 2, Cols: 3, Spacing: 2.5");
    
    // Calculate expected positions manually
    let grid_width = (3.0 - 1.0) * 2.5;  // 5.0
    let grid_depth = (2.0 - 1.0) * 2.5;  // 2.5
    let start_x = -grid_width / 2.0;     // -2.5
    let start_z = -grid_depth / 2.0;     // -1.25
    
    println!("Grid dimensions: width={:.1}, depth={:.1}", grid_width, grid_depth);
    println!("Start position: x={:.2}, z={:.2}", start_x, start_z);
    
    println!("\nExpected positions:");
    for row in 0..2 {
        for col in 0..3 {
            let x = start_x + col as f32 * 2.5;
            let z = start_z + row as f32 * 2.5;
            let index = row * 3 + col;
            println!("  CartPole {} (row={}, col={}): x={:.2}, z={:.2}", index, row, col, x, z);
        }
    }
    
    println!("\nActual positions:");
    for (i, cp) in grid.cartpoles.iter().enumerate() {
        let pos = sim.boxes[cp.cart_idx].pos;
        println!("  CartPole {}: x={:.2}, y={:.2}, z={:.2}", i, pos.x, pos.y, pos.z);
    }
    
    // Check for exact overlaps
    println!("\nOverlap check:");
    for i in 0..grid.cartpoles.len() {
        for j in (i+1)..grid.cartpoles.len() {
            let pos_i = sim.boxes[grid.cartpoles[i].cart_idx].pos;
            let pos_j = sim.boxes[grid.cartpoles[j].cart_idx].pos;
            
            let distance = (pos_i - pos_j).length();
            if distance < 0.001 {
                println!("  EXACT OVERLAP: CartPoles {} and {} both at ({:.3}, {:.3}, {:.3})", 
                         i, j, pos_i.x, pos_i.y, pos_i.z);
            } else if distance < 1.0 {
                println!("  Close: CartPoles {} and {} distance={:.3}", i, j, distance);
            }
        }
    }
}