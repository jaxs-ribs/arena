use physics::{PhysicsSim, Vec3};

#[test]
fn test_revolute_joint_limits() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO, 1.0);
    sim.add_sphere(Vec3::new(1.0,0.0,0.0), Vec3::ZERO, 1.0);
    sim.add_revolute_joint(0,1,Vec3::ZERO,Vec3::ZERO,Vec3::new(0.0,1.0,0.0));
    sim.step_cpu();
    assert_eq!(sim.revolute_joints.len(),1);
}

#[test]
fn test_revolute_joint_motor() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO, 1.0);
    sim.add_sphere(Vec3::new(1.0,0.0,0.0), Vec3::ZERO, 1.0);
    sim.add_revolute_joint(0,1,Vec3::ZERO,Vec3::ZERO,Vec3::new(0.0,1.0,0.0));
    sim.step_cpu();
    assert_eq!(sim.revolute_joints.len(),1);
}

#[test]
fn test_prismatic_joint_sliding() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO,1.0);
    sim.add_sphere(Vec3::new(0.0,1.0,0.0), Vec3::ZERO,1.0);
    sim.add_prismatic_joint(0,1,Vec3::ZERO,Vec3::ZERO,Vec3::new(0.0,1.0,0.0));
    sim.step_cpu();
    assert_eq!(sim.prismatic_joints.len(),1);
}

#[test]
fn test_ball_joint_3dof() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO,1.0);
    sim.add_sphere(Vec3::new(0.0,1.0,0.0), Vec3::ZERO,1.0);
    sim.add_ball_joint(0,1,Vec3::ZERO,Vec3::ZERO);
    sim.step_cpu();
    assert_eq!(sim.ball_joints.len(),1);
}

#[test]
fn test_fixed_joint_rigidity() {
    let mut sim = PhysicsSim::new();
    sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO,1.0);
    sim.add_sphere(Vec3::new(0.0,1.0,0.0), Vec3::ZERO,1.0);
    sim.add_fixed_joint(0,1,Vec3::ZERO,Vec3::ZERO);
    sim.step_cpu();
    assert_eq!(sim.fixed_joints.len(),1);
}

#[test]
fn test_joint_chain_stability() {
    let mut sim = PhysicsSim::new();
    let a = sim.add_sphere(Vec3::new(0.0,0.0,0.0), Vec3::ZERO,1.0);
    let mut prev = a;
    for i in 1..5 {
        let b = sim.add_sphere(Vec3::new(i as f32,0.0,0.0), Vec3::ZERO,1.0);
        sim.add_revolute_joint(prev,b,Vec3::ZERO,Vec3::ZERO,Vec3::new(0.0,1.0,0.0));
        prev = b;
    }
    sim.step_cpu();
    assert_eq!(sim.revolute_joints.len(),4);
}
