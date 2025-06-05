use criterion::{criterion_group, criterion_main, Criterion};
use physics::{PhysicsSim, Sphere, Vec3, Joint};

fn bench_scene_run(c: &mut Criterion) {
    c.bench_function("scene_run", |b| {
        b.iter(|| {
            let mut sim = PhysicsSim::new_single_sphere(1.0);
            let num = 10u32;
            for i in 1..num {
                sim.spheres.push(Sphere {
                    pos: Vec3::new(i as f32, 1.0, 0.0),
                    vel: Vec3::new(0.0, 0.0, 0.0),
                });
                sim.joints.push(Joint {
                    body_a: i - 1,
                    body_b: i,
                    rest_length: 1.0,
                    _padding: 0,
                });
            }
            sim.run(0.01, 10).unwrap();
        })
    });
}

criterion_group!(benches, bench_scene_run);
criterion_main!(benches);
