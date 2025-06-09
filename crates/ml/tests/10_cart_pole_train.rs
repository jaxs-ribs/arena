use ml::rl::StickBalancePpoTrainer;
use ml::StickBalanceEnv;
use ml::Env;

#[test]
#[ignore]
fn ppo_learns_to_balance_cart_pole() {
    let mut trainer = StickBalancePpoTrainer::new(0);
    let mut best = 0.0f32;
    for _ in 0..500 { // 500 training iterations
        let reward = trainer.step();
        if reward > best {
            best = reward;
        }
        if best > 50.0 { // close to perfect (64)
            break;
        }
    }
    assert!(best > 50.0, "best {best}");

    // Verify that the trained policy keeps the pole balanced for many steps.
    let mut env = StickBalanceEnv::new();
    let mut obs = env.reset();
    for i in 0..100 {
        let action = trainer.act(&obs);
        let (o, _r, done) = env.step(action);
        obs = o;
        if done {
            panic!("fell over after {} steps", i);
        }
    }
}
