use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn debug_cylinder_visual_representation() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        pole_radius: 0.1,
        initial_angle: 0.0, // Vertical
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Get all the key positions
    let cart = &sim.boxes[cartpole.cart_idx];
    let pole = &sim.cylinders[cartpole.pole_idx];
    
    println!("=== VISUAL DEBUG ===");
    println!("Cart:");
    println!("  Position: {:?}", cart.pos);
    println!("  Half extents: {:?}", cart.half_extents);
    println!("  Top Y: {}", cart.pos.y + cart.half_extents.y);
    println!("  Bottom Y: {}", cart.pos.y - cart.half_extents.y);
    
    println!("\nPole:");
    println!("  Position (center): {:?}", pole.pos);
    println!("  Half height: {}", pole.half_height);
    println!("  Full height: {}", pole.half_height * 2.0);
    println!("  Radius: {}", pole.radius);
    println!("  Top Y: {}", pole.pos.y + pole.half_height);
    println!("  Bottom Y: {}", pole.pos.y - pole.half_height);
    println!("  Orientation: {:?}", pole.orientation);
    
    println!("\nJoint:");
    let joint_y = cart.pos.y + cart.half_extents.y;
    println!("  Joint Y (top of cart): {}", joint_y);
    println!("  Pole bottom - Joint Y: {}", (pole.pos.y - pole.half_height) - joint_y);
    
    println!("\nVisual Checks:");
    println!("  Is pole bottom at joint? {}", ((pole.pos.y - pole.half_height) - joint_y).abs() < 0.01);
    println!("  Is pole standing UP? {}", pole.pos.y > joint_y);
    println!("  Pole center above cart top? {}", pole.pos.y > (cart.pos.y + cart.half_extents.y));
    
    // What the renderer sees
    println!("\nRenderer Data (after conversion):");
    println!("  Cylinder pos: {:?}", pole.pos);
    println!("  Cylinder height (full): {}", pole.half_height * 2.0);
    println!("  Cylinder orientation: {:?}", pole.orientation);
    
    // Visual bounding box
    let visual_top = pole.pos.y + pole.half_height;
    let visual_bottom = pole.pos.y - pole.half_height;
    println!("\nVisual Bounding Box:");
    println!("  Top: {}", visual_top);
    println!("  Bottom: {}", visual_bottom);
    println!("  Should touch cart at Y={}", joint_y);
}