/*
use criterion::{criterion_group, criterion_main, Criterion};
use physics::PhysicsSim;

fn bench_1000_spheres(c: &mut Criterion) {
    let mut sim = PhysicsSim::new_single_sphere(10.0);
    // initialize many spheres if available
    c.bench_function("sphere_step", |b| b.iter(|| sim.step_gpu().unwrap()));
}

criterion_group!(benches, bench_1000_spheres);
criterion_main!(benches);
*/
