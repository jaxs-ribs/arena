use physics::{PhysicsSim, types::{Vec3, BodyType}};

fn main() {
    let mut sim = PhysicsSim::new();
    
    // Create a simple cylinder
    let cylinder_idx = sim.add_cylinder_with_type(
        Vec3::new(0.0, 0.0, 0.0), // position
        0.1,  // radius
        1.0,  // half_height
        Vec3::ZERO, // velocity
        BodyType::Dynamic
    );
    
    // Manually set angular velocity around Z-axis
    sim.cylinders[cylinder_idx].angular_vel.z = 1.0; // 1 rad/s
    
    println!("=== Initial State ===");
    println!("Position: {:?}", sim.cylinders[cylinder_idx].pos);
    println!("Orientation: {:?}", sim.cylinders[cylinder_idx].orientation);
    println!("Angular Velocity: {:?}", sim.cylinders[cylinder_idx].angular_vel);
    
    // Step the simulation multiple times
    for step in 1..=10 {
        sim.step_cpu();
        
        println!("\n=== After Step {} ===", step);
        println!("Position: {:?}", sim.cylinders[cylinder_idx].pos);
        println!("Orientation: {:?}", sim.cylinders[cylinder_idx].orientation);
        println!("Angular Velocity: {:?}", sim.cylinders[cylinder_idx].angular_vel);
        
        // Calculate angle from quaternion
        let q = sim.cylinders[cylinder_idx].orientation;
        let angle_z = 2.0 * (q[3] * q[2] + q[0] * q[1]).atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
        println!("Z-axis rotation from quaternion: {:.3} rad ({:.1}°)", angle_z, angle_z.to_degrees());
    }
}