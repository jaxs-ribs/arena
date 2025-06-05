use physics::{Cube, Vec3};

#[test]
fn cube_creation() {
    let c = Cube::new(2.0, 3.0);
    assert_eq!(c.side, 2.0);
    assert_eq!(c.mass, 3.0);
}

#[test]
fn cube_sdf_basic() {
    let cube = Cube::new(2.0, 1.0);
    let eps = 1e-6f32;
    assert!((cube.sdf(Vec3::new(0.0, 0.0, 0.0)) + 1.0).abs() < eps);
    assert!(cube.sdf(Vec3::new(1.0, 0.0, 0.0)).abs() < eps);
    assert!((cube.sdf(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < eps);
}
