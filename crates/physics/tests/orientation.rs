use physics::PhysicsSim;
use physics::Vec3;

#[test]
fn torque_rotates_sphere() {
    let mut sim = PhysicsSim::new_single_sphere(0.0);
    sim.spheres[0].angular_vel = Vec3::new(0.0, 0.0, 1.0);
    let _ = sim.run(0.1, 1).unwrap();
    assert!((sim.spheres[0].orientation[2] - 0.05).abs() < 1e-5);
}
