use physics::{PhysicsSim, Vec3};

#[test]
fn collision_matrix_runs() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
    sim.add_box(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO);
    sim.add_cylinder(Vec3::new(4.0, 0.0, 0.0), 1.0, 2.0, Vec3::ZERO);
    sim.add_plane(Vec3::new(0.0, 1.0, 0.0), -5.0);

    // A single CPU step should run without panicking even with placeholder collisions.
    sim.step_cpu();
}
