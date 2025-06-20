use physics::{PhysicsSim, CartPoleGrid, CartPoleConfig, Vec3};

fn main() {
    println!("=== Debugging Missing Cylinders ===");
    
    let mut sim = PhysicsSim::new();
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    
    // Add ground plane
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, physics::Vec2::new(15.0, 15.0));
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        cart_mass: 1.0,
        pole_length: 1.5,
        pole_radius: 0.05,
        pole_mass: 0.1,
        initial_angle: 0.05,
        force_magnitude: 10.0,
        failure_angle: 0.5,
        position_limit: 3.0,
    };
    
    println!("Before creating CartPoles:");
    println!("  Boxes: {}", sim.boxes.len());
    println!("  Cylinders: {}", sim.cylinders.len());
    println!("  Revolute joints: {}", sim.revolute_joints.len());
    
    let grid = CartPoleGrid::new(&mut sim, 2, 3, 2.0, config.clone());
    
    println!("\nAfter creating CartPoles:");
    println!("  Boxes: {}", sim.boxes.len());
    println!("  Cylinders: {}", sim.cylinders.len());
    println!("  Revolute joints: {}", sim.revolute_joints.len());
    
    // Check each cartpole
    for (i, cartpole) in grid.cartpoles.iter().enumerate() {
        println!("\nCartPole {}:", i);
        println!("  Cart index: {}", cartpole.cart_idx);
        println!("  Pole index: {}", cartpole.pole_idx);
        println!("  Joint index: {}", cartpole.joint_idx);
        
        if cartpole.cart_idx < sim.boxes.len() {
            let cart = &sim.boxes[cartpole.cart_idx];
            println!("  Cart pos: ({:.2}, {:.2}, {:.2})", cart.pos.x, cart.pos.y, cart.pos.z);
            println!("  Cart half_extents: ({:.2}, {:.2}, {:.2})", 
                     cart.half_extents.x, cart.half_extents.y, cart.half_extents.z);
        } else {
            println!("  ERROR: Cart index {} out of bounds!", cartpole.cart_idx);
        }
        
        if cartpole.pole_idx < sim.cylinders.len() {
            let pole = &sim.cylinders[cartpole.pole_idx];
            println!("  Pole pos: ({:.2}, {:.2}, {:.2})", pole.pos.x, pole.pos.y, pole.pos.z);
            println!("  Pole radius: {:.2}, half_height: {:.2}", pole.radius, pole.half_height);
        } else {
            println!("  ERROR: Pole index {} out of bounds!", cartpole.pole_idx);
        }
        
        if cartpole.joint_idx < sim.revolute_joints.len() {
            let joint = &sim.revolute_joints[cartpole.joint_idx];
            println!("  Joint connects: body_a={}, body_b={}", joint.body_a, joint.body_b);
            println!("  Anchor A: ({:.2}, {:.2}, {:.2})", joint.anchor_a.x, joint.anchor_a.y, joint.anchor_a.z);
            println!("  Anchor B: ({:.2}, {:.2}, {:.2})", joint.anchor_b.x, joint.anchor_b.y, joint.anchor_b.z);
        } else {
            println!("  ERROR: Joint index {} out of bounds!", cartpole.joint_idx);
        }
    }
    
    // Check cylinder visibility
    println!("\nCylinder details:");
    for (i, cylinder) in sim.cylinders.iter().enumerate() {
        println!("  Cylinder {}: pos=({:.2}, {:.2}, {:.2}), r={:.2}, h={:.2}", 
                 i, cylinder.pos.x, cylinder.pos.y, cylinder.pos.z,
                 cylinder.radius, cylinder.half_height);
    }
}