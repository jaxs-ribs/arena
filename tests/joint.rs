use physics::{PhysicsSim, Sphere, Vec3, Joint};

#[test]
fn distance_joint_moves_bodies_toward_rest_length() {
    let mut sim = PhysicsSim::new_single_sphere(0.0);
    sim.spheres.push(Sphere::new(
        Vec3::new(1.5, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
    ));
    sim.params.forces.push([0.0, 0.0]);
    sim.joints.push(Joint { body_a: 0, body_b: 1, rest_length: 1.0, _padding: 0 });
    let _ = sim.run(0.0, 1).unwrap();
    let dx = sim.spheres[1].pos.x - sim.spheres[0].pos.x;
    let dy = sim.spheres[1].pos.y - sim.spheres[0].pos.y;
    let dz = sim.spheres[1].pos.z - sim.spheres[0].pos.z;
    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
    assert!((dist - 1.0).abs() < 1e-5);
}
