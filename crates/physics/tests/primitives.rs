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
    sim.run_cpu(0.01, 200);
    let sphere_y = sim.spheres[0].pos.y;
    let box_y = sim.boxes[0].pos.y;
    let cyl_y = sim.cylinders[0].pos.y;
    assert!(sphere_y >= 1.0 - 1e-3);
    assert!(box_y >= 0.5 - 1e-3);
    assert!(cyl_y >= 0.5 - 1e-3);
}
