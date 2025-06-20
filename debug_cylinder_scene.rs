use physics::{PhysicsSim, Vec3};

fn main() {
    println!("=== Creating Debug Cylinder Scene ===");
    
    let mut sim = PhysicsSim::new();
    
    // Add ground plane
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, physics::Vec2::new(15.0, 15.0));
    
    // Add a large cylinder at origin
    let cyl1 = sim.add_cylinder(
        Vec3::new(0.0, 2.0, 0.0),  // Center at y=2
        0.5,                       // Radius 0.5m (large)
        2.0,                       // Half-height 2m (4m tall total)
        Vec3::ZERO
    );
    println!("Added large cylinder at origin: idx={}", cyl1);
    
    // Add a box for reference
    let box1 = sim.add_box(
        Vec3::new(3.0, 1.0, 0.0),  // Offset to the right
        Vec3::new(0.5, 0.5, 0.5),  // 1m cube
        Vec3::ZERO
    );
    println!("Added reference box: idx={}", box1);
    
    // Add a sphere for reference
    let sph1 = sim.add_sphere(
        Vec3::new(-3.0, 1.0, 0.0), // Offset to the left
        Vec3::ZERO,
        0.5
    );
    println!("Added reference sphere: idx={}", sph1);
    
    println!("\nScene contains:");
    println!("  {} cylinders", sim.cylinders.len());
    println!("  {} boxes", sim.boxes.len());
    println!("  {} spheres", sim.spheres.len());
    println!("  {} planes", sim.planes.len());
    
    // Write scene to a file that app.rs can load
    use std::fs::File;
    use std::io::Write;
    
    let mut f = File::create("debug_scene.txt").unwrap();
    writeln!(f, "cylinder 0.0 2.0 0.0 0.5 2.0").unwrap();
    writeln!(f, "box 3.0 1.0 0.0 0.5 0.5 0.5").unwrap();
    writeln!(f, "sphere -3.0 1.0 0.0 0.5").unwrap();
    
    println!("\nDebug scene saved to debug_scene.txt");
    println!("To test: modify app.rs to load this scene instead of CartPoles");
}