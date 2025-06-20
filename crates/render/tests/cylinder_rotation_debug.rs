use physics::{PhysicsSim, cartpole::{CartPole, CartPoleConfig}};
use physics::types::Vec3;

#[test]
fn test_cylinder_rotation_coordinate_system() {
    println!("🧪 Testing cylinder rotation coordinate system...");
    
    // Create physics sim with CartPole
    let mut sim = PhysicsSim::new();
    let config = CartPoleConfig {
        initial_angle: 0.0, // Start perfectly vertical
        ..Default::default()
    };
    let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);
    
    println!("🔵 Initial state (vertical pole):");
    println!("    Position: ({:.3}, {:.3}, {:.3})", 
             sim.cylinders[cartpole.pole_idx].pos.x,
             sim.cylinders[cartpole.pole_idx].pos.y,
             sim.cylinders[cartpole.pole_idx].pos.z);
    println!("    Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]",
             sim.cylinders[cartpole.pole_idx].orientation[0],
             sim.cylinders[cartpole.pole_idx].orientation[1],
             sim.cylinders[cartpole.pole_idx].orientation[2],
             sim.cylinders[cartpole.pole_idx].orientation[3]);
    
    // Now create one with 90 degree angle
    let mut sim2 = PhysicsSim::new();
    let config2 = CartPoleConfig {
        initial_angle: std::f32::consts::PI / 2.0, // 90 degrees
        ..Default::default()
    };
    let cartpole2 = CartPole::new(&mut sim2, Vec3::ZERO, config2);
    
    println!("\n🔵 90° tilted state (horizontal pole):");
    println!("    Position: ({:.3}, {:.3}, {:.3})", 
             sim2.cylinders[cartpole2.pole_idx].pos.x,
             sim2.cylinders[cartpole2.pole_idx].pos.y,
             sim2.cylinders[cartpole2.pole_idx].pos.z);
    println!("    Orientation: [{:.3}, {:.3}, {:.3}, {:.3}]",
             sim2.cylinders[cartpole2.pole_idx].orientation[0],
             sim2.cylinders[cartpole2.pole_idx].orientation[1],
             sim2.cylinders[cartpole2.pole_idx].orientation[2],
             sim2.cylinders[cartpole2.pole_idx].orientation[3]);
    
    // Calculate what the pole vector should be
    let cart_pos = sim2.boxes[cartpole2.cart_idx].pos;
    let pole_pos = sim2.cylinders[cartpole2.pole_idx].pos;
    let joint_pos = cart_pos + Vec3::new(0.0, 0.1, 0.0); // Use default cart_size.y
    let pole_vector = pole_pos - joint_pos;
    
    println!("\n🔵 Pole vector from joint to center:");
    println!("    Vector: ({:.3}, {:.3}, {:.3})", pole_vector.x, pole_vector.y, pole_vector.z);
    println!("    Expected for 90°: (1.0, 0.0, 0.0)");
    
    // Test the shader's expectation
    println!("\n🔵 Shader expectation:");
    println!("    - Cylinder is Y-aligned in local space");
    println!("    - Identity quaternion [0,0,0,1] = vertical cylinder");
    println!("    - Rotation around Z should tilt cylinder in X-Y plane");
}

#[test]
fn test_quaternion_rotation_visual() {
    println!("🧪 Testing quaternion rotation visualization...");
    
    // Test different quaternions to understand the rotation
    let test_quats = [
        ([0.0, 0.0, 0.0, 1.0], "Identity (no rotation)"),
        ([0.0, 0.0, 0.707, 0.707], "90° around Z-axis"),
        ([0.707, 0.0, 0.0, 0.707], "90° around X-axis"),
        ([0.0, 0.707, 0.0, 0.707], "90° around Y-axis"),
    ];
    
    for (quat, desc) in test_quats {
        println!("\n🔵 {}: {:?}", desc, quat);
        
        // Apply rotation to a Y-aligned vector (like our cylinder)
        let y_vec = [0.0, 1.0, 0.0]; // Y-up vector
        let rotated = apply_quaternion_rotation(y_vec, quat);
        
        println!("    Y-vector [0,1,0] becomes: [{:.3}, {:.3}, {:.3}]", 
                 rotated[0], rotated[1], rotated[2]);
    }
}

fn apply_quaternion_rotation(v: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    // Quaternion rotation formula: v' = q * v * q^-1
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    
    let vx = v[0];
    let vy = v[1];
    let vz = v[2];
    
    // Using the optimized formula: v' = v + 2 * q_xyz × (q_xyz × v + q_w * v)
    let cross1_x = qy * vz - qz * vy;
    let cross1_y = qz * vx - qx * vz;
    let cross1_z = qx * vy - qy * vx;
    
    let temp_x = cross1_x + qw * vx;
    let temp_y = cross1_y + qw * vy;
    let temp_z = cross1_z + qw * vz;
    
    let cross2_x = qy * temp_z - qz * temp_y;
    let cross2_y = qz * temp_x - qx * temp_z;
    let cross2_z = qx * temp_y - qy * temp_x;
    
    [
        vx + 2.0 * cross2_x,
        vy + 2.0 * cross2_y,
        vz + 2.0 * cross2_z,
    ]
}