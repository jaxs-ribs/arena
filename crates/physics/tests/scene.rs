use physics::{PhysicsSim, Sphere, Vec3, Joint, JOINT_TYPE_DISTANCE};

#[test]
fn chain_of_spheres_runs_stably() {
    let mut sim = PhysicsSim::new_single_sphere(1.0);
    let num = 10u32;
    for i in 1..num {
        sim.spheres.push(Sphere::new(
            Vec3::new(i as f32, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
        sim.params.forces.push([0.0, 0.0]);
        sim.joints.push(Joint {
            body_a: i - 1,
            body_b: i,
            joint_type: JOINT_TYPE_DISTANCE,
            rest_length: 1.0,
            local_anchor_a: Vec3::new(0.0, 0.0, 0.0),
            local_anchor_b: Vec3::new(0.0, 0.0, 0.0),
            local_axis_a: Vec3::new(0.0, 0.0, 0.0),
            local_axis_b: Vec3::new(0.0, 0.0, 0.0),
        });
    }

    let _ = sim.run(0.01, 10).unwrap();

    for s in &sim.spheres {
        assert!(s.pos.y >= 1.0);
    }

    // Check chain length approximately maintained
    let start = sim.spheres.first().unwrap().pos;
    let end = sim.spheres.last().unwrap().pos;
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let dz = end.z - start.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!((dist - ((num - 1) as f32)).abs() < 0.5);
}
