use physics::{PhysicsSim, Vec3};

#[test]
fn add_sphere_updates_force_len() {
    let mut sim = PhysicsSim::new();
    assert_eq!(sim.spheres.len(), 0);
    let idx = sim.add_sphere(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(idx, 0);
    assert_eq!(sim.spheres.len(), 1);
    assert_eq!(sim.params.forces.len(), 1);
}

#[test]
fn set_force_affects_simulation() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    sim.set_force(0, [1.0, 0.0]);
    let _ = sim.run(0.1, 1).unwrap();
    assert!(sim.spheres[0].pos.x > 0.0);
}
