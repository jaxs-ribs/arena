use rl::env::SimpleEnv;
use rl::ppo::PpoAgent;
use ml::Tensor;

#[test]
fn ppo_training_reduces_error() {
    let env = SimpleEnv::new(1.0);
    let mut state = env.reset();
    let mut agent = PpoAgent::new(0.01);
    let obs = Tensor::from_vec(vec![1], vec![0.0]);
    let mut action = agent.act(&obs);
    let (_, reward, _) = env.step(&mut state, action);
    let mut prev_loss = -reward;
    let start_loss = prev_loss;
    for _ in 0..50 {
        let mut advantage = env.target - action;
        if advantage > 10.0 { advantage = 10.0; }
        if advantage < -10.0 { advantage = -10.0; }
        agent.update(&obs, action, advantage);
        action = agent.act(&obs);
        let (_, reward, _) = env.step(&mut state, action);
        prev_loss = -reward;
    }
    eprintln!("start_loss {start_loss} final_loss {prev_loss}");
    assert!(prev_loss < start_loss, "agent should learn to reduce loss");
}
