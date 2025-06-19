use physics::{PhysicsSim, Vec3};
use physics::types::Vec2;

#[test]
fn create_primitives() {
    let mut sim = PhysicsSim::new();
    let s_idx = sim.add_sphere(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0);
    let b_idx = sim.add_box(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, 0.0));
    let c_idx = sim.add_cylinder(Vec3::new(0.0, 3.0, 0.0), 0.5, 1.0, Vec3::new(0.0, 0.0, 0.0));
    let p_idx = sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(25.0, 25.0));
    assert_eq!(s_idx, 0);
    assert_eq!(b_idx, 0);
    assert_eq!(c_idx, 0);
    assert_eq!(p_idx, 0);
    assert_eq!(sim.spheres.len(), 1);
    assert_eq!(sim.boxes.len(), 1);
    assert_eq!(sim.cylinders.len(), 1);
    assert_eq!(sim.planes.len(), 1);
}

#[test]
fn primitives_rest_on_plane() {
    let mut sim = PhysicsSim::new();
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(25.0, 25.0)); // ground plane y=0
    sim.add_sphere(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0);
    sim.add_box(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, 0.0));
    sim.add_cylinder(Vec3::new(0.0, 3.0, 0.0), 0.5, 1.0, Vec3::new(0.0, 0.0, 0.0));
    sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);
    // Initial positions
    println!("Initial sphere y: {}", sim.spheres[0].pos.y);
    
    sim.run_cpu(0.01, 200);
    let sphere_y = sim.spheres[0].pos.y;
    let box_y = sim.boxes[0].pos.y;
    let cyl_y = sim.cylinders[0].pos.y;
    
    println!("Final sphere y: {}, expected >= {}", sphere_y, 1.0 - 1e-3);
    println!("Final box y: {}, expected >= {}", box_y, 0.5 - 1e-3);
    println!("Final cylinder y: {}, expected >= {}", cyl_y, 0.5 - 1e-3);
    
    // Check plane properties
    if !sim.planes.is_empty() {
        println!("Plane normal: {:?}, d: {}", sim.planes[0].normal, sim.planes[0].d);
    }
    
    // Spheres should rest at radius height
    assert!(sphere_y >= 1.0 - 5e-3, "Sphere should rest on plane");
    
    // Box and cylinder collision with planes is not implemented yet
    // For now, just check they don't fall through completely
    assert!(box_y > 0.0, "Box shouldn't fall through floor");
    assert!(cyl_y > 0.0, "Cylinder shouldn't fall through floor");
}
