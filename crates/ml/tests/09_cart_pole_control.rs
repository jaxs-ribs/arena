use ml::StickBalanceEnv;
use ml::Env;

#[test]
fn cart_pole_base_moves() {
    let mut env = StickBalanceEnv::new();
    let mut obs = env.reset();
    assert_eq!(obs.len(), 2);
    for _ in 0..10 {
        let (o, _r, _d) = env.step(5.0);
        obs = o;
    }
    // the base position is the first element of the observation
    assert!(obs[0] > 0.0, "cart should move right with positive force");
}
