use ml::rl::*;

#[test]
#[ignore]
fn ppo_improves_reward() {
    let mut trainer = SpherePpoTrainer::new(0);
    let mut best = f32::MIN;
    for _ in 0..5_000 {
        best = best.max(trainer.step());
    }
    assert!(best > 4.0, "best {best}");
}
