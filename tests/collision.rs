use physics::{PhysicsSim, Vec3};

#[test]
fn sphere_resolves_floor_contact() {
    let mut sim = PhysicsSim::new_single_sphere(-0.1);
    // One step with small dt
    let _ = sim.run(0.01, 1).unwrap();
    assert!(sim.spheres[0].pos.y >= 0.0);
}
