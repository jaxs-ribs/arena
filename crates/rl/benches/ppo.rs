/*
use criterion::{criterion_group, criterion_main, Criterion};
use rl::env::SimpleEnv;
use rl::ppo::PpoAgent;
use ml::Tensor;
use sysinfo::System;

fn bench_ppo_train_step(c: &mut Criterion) {
    let env = SimpleEnv::new(1.0);
    let obs = Tensor::from_vec(vec![1], vec![0.0]);
    c.bench_function("ppo_train_step", |b| {
        let mut agent = PpoAgent::new(0.01);
        b.iter(|| {
            let mut state = env.reset();
            let action = agent.act(&obs);
            let advantage = env.target - action;
            agent.update(&obs, action, advantage);
        });
    });

    // Print hardware stats
    let mut sys = System::new_all();
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand())
        .unwrap_or("unknown");
    let cores = System::physical_core_count().unwrap_or(sys.cpus().len());
    let mem_mb = sys.total_memory() / 1024;
    println!("Hardware: {} with {} cores, {} MB RAM", cpu_brand, cores, mem_mb);
}

criterion_group!(benches, bench_ppo_train_step);
criterion_main!(benches);
*/
