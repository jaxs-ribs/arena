use physics::{PhysicsSim, Sphere, Vec3};

#[test]
fn per_body_forces_move_spheres_independently() {
    let mut sim = PhysicsSim::new_single_sphere(0.0);
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    sim.spheres.push(Sphere::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
    ));
    sim.params.forces.push([ -1.0, 0.0 ]);
    sim.params.forces[0] = [1.0, 0.0];

    let _ = sim.run(0.1, 1).unwrap();

    let pos0 = sim.spheres[0].pos.x;
    let pos1 = sim.spheres[1].pos.x;
    assert!(pos0 > 0.0);
    assert!(pos1 < 0.0);
}
