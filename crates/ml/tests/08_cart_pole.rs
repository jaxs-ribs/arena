use ml::StickBalanceEnv;
use ml::env::Env;

/// Runs the cart-pole (stick balance) environment with zero control.
/// Starting from a small angle offset, the pole should fall over
/// without intervention and the episode should terminate.
#[test]
fn cart_pole_falls_without_control() {
    let mut env = StickBalanceEnv::new();
    let _ = env.reset_with_angle(0.05);
    let mut done = false;
    for _ in 0..200 {
        let (_obs, _r, d) = env.step(0.0);
        if d {
            done = true;
            break;
        }
    }
    assert!(done, "episode should end once the pole falls over");
}
