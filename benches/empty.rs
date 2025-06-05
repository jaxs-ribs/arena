use criterion::{criterion_group, criterion_main, Criterion};

fn bench_dummy(c: &mut Criterion) {
    c.bench_function("noop_bench", |b| b.iter(|| 2 + 2));
}

criterion_group!(benches, bench_dummy);
criterion_main!(benches); 