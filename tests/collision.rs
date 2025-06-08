use physics::{PhysicsSim, Vec3};

#[test]
fn sphere_resolves_floor_contact() {
    let mut sim = PhysicsSim::new_single_sphere(-0.1);
    // One step with small dt
    let _ = sim.run(0.01, 1).unwrap();
    assert!(sim.spheres[0].pos.y >= 0.0);
}

#[test]
fn two_spheres_collide_in_free_fall() {
    use physics::{Sphere};

    let mut sim = PhysicsSim::new_single_sphere(5.0);
    sim.spheres[0].vel = Vec3::new(0.0, -2.0, 0.0);
    sim.spheres.push(Sphere { pos: Vec3::new(0.0, 2.0, 0.0), vel: Vec3::new(0.0, 0.0, 0.0) });

    let _ = sim.run(0.01, 51).unwrap();

    let dx = sim.spheres[0].pos.x - sim.spheres[1].pos.x;
    let dy = sim.spheres[0].pos.y - sim.spheres[1].pos.y;
    let dz = sim.spheres[0].pos.z - sim.spheres[1].pos.z;
    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
    assert!(dist >= 2.0 - 1e-3);
}
