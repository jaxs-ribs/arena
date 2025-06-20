use physics::{PhysicsSim, Vec3};

fn main() {
    println!("=== Debugging Joint Creation ===");
    
    let mut sim = PhysicsSim::new();
    
    // Add a box (cart)
    let cart_idx = sim.add_box(Vec3::ZERO, Vec3::new(0.4, 0.2, 0.2), Vec3::ZERO);
    println!("Created cart at index: {}", cart_idx);
    
    // Add a cylinder (pole)  
    let pole_idx = sim.add_cylinder(Vec3::new(0.0, 1.0, 0.0), 0.05, 0.75, Vec3::ZERO);
    println!("Created pole at index: {}", pole_idx);
    
    // Add revolute joint
    println!("\nCalling add_revolute_joint with:");
    println!("  body_a_type: 0 (box), body_a_index: {}", cart_idx);
    println!("  body_b_type: 2 (cylinder), body_b_index: {}", pole_idx);
    
    let joint_idx = sim.add_revolute_joint(
        0, cart_idx as u32,  // Box type, cart index
        2, pole_idx as u32,  // Cylinder type, pole index
        Vec3::new(0.0, 0.4, 0.0), // Joint position
        Vec3::new(0.0, 0.0, 1.0)  // Z axis
    );
    
    println!("\nCreated joint at index: {}", joint_idx);
    
    // Check the joint
    let joint = &sim.revolute_joints[joint_idx];
    println!("\nJoint details:");
    println!("  body_a: {}", joint.body_a);
    println!("  body_b: {}", joint.body_b);
    println!("  anchor_a: ({:.2}, {:.2}, {:.2})", joint.anchor_a.x, joint.anchor_a.y, joint.anchor_a.z);
    println!("  anchor_b: ({:.2}, {:.2}, {:.2})", joint.anchor_b.x, joint.anchor_b.y, joint.anchor_b.z);
}