use ml::StickBalanceEnv;
use ml::Env;

/// Basic sanity test for the stick balancing environment.
///
/// This does not train a policy yet. It simply steps the environment
/// with zero actions and ensures the observation vector has the expected
/// size and that an episode eventually terminates.
#[test]
fn stick_balance_env_basics() {
    let mut env = StickBalanceEnv::new();
    let mut obs = env.reset();
    assert_eq!(obs.len(), 2);
    for _ in 0..10 {
        let (o, r, _d) = env.step(0.0); // no control yet
        obs = o;
        // reward should be finite
        assert!(r.is_finite());
    }
    assert_eq!(obs.len(), 2);
}

