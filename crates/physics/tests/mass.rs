use physics::{PhysicsSim, Sphere, Vec3};

#[test]
fn heavier_sphere_accelerates_less() {
    let mut sim = PhysicsSim::new_single_sphere(0.0);
    sim.params.gravity = Vec3::new(0.0, 0.0, 0.0);
    let mut heavy = Sphere::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
    heavy.mass = 2.0;
    heavy.inv_inertia = 1.0 / (0.4 * heavy.mass);
    sim.spheres.push(heavy);
    sim.params.forces.push([1.0, 0.0]);
    sim.params.forces[0] = [1.0, 0.0];

    let _ = sim.run(0.1, 1).unwrap();
    let light_vx = sim.spheres[0].vel.x;
    let heavy_vx = sim.spheres[1].vel.x;
    assert!(light_vx > heavy_vx);
}
