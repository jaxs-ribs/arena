use crate::{nn::Dense, optim::Adam, recorder::Recorder, tape::Tape, Tensor};
use std::collections::HashMap;

pub trait Env {
    fn step(&mut self, action: f32) -> (Vec<f32>, f32, bool);
    fn reset(&mut self) -> Vec<f32>;
    fn obs_size(&self) -> usize;
    fn action_size(&self) -> usize;
}

pub struct RollingSphereEnv {
    pos_x: f32,
}

impl RollingSphereEnv {
    pub fn new() -> Self {
        Self { pos_x: 0.0 }
    }
}

impl Env for RollingSphereEnv {
    fn step(&mut self, action: f32) -> (Vec<f32>, f32, bool) {
        let force = action.max(-10.0).min(10.0);
        let old_pos_x = self.pos_x;
        self.pos_x += force * 0.02;
        let reward = self.pos_x - old_pos_x;
        let done = self.pos_x.abs() > 5.0;
        (vec![self.pos_x], reward, done)
    }

    fn reset(&mut self) -> Vec<f32> {
        self.pos_x = 0.0;
        vec![self.pos_x]
    }

    fn obs_size(&self) -> usize { 1 }
    fn action_size(&self) -> usize { 1 }
}

pub struct PolicyNet {
    l1: Dense,
    l2: Dense,
}

impl PolicyNet {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            l1: Dense::random(in_dim, hidden_dim, 0),
            l2: Dense::random(hidden_dim, out_dim, 0),
        }
    }

    pub fn forward(&self, x: &Tensor, recorder: &mut impl Recorder) -> Tensor {
        let (x, _) = self.l1.forward(x, recorder);
        let x = x.tanh(recorder);
        let (x, _) = self.l2.forward(&x, recorder);
        x
    }

    pub fn get_params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.l1.w, &mut self.l1.b, &mut self.l2.w, &mut self.l2.b]
    }
}

pub struct ValueNet {
    l1: Dense,
    l2: Dense,
}

impl ValueNet {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            l1: Dense::random(in_dim, hidden_dim, 0),
            l2: Dense::random(hidden_dim, out_dim, 0),
        }
    }

    pub fn forward(&self, x: &Tensor, recorder: &mut impl Recorder) -> Tensor {
        let (x, _) = self.l1.forward(x, recorder);
        let x = x.tanh(recorder);
        let (x, _) = self.l2.forward(&x, recorder);
        x
    }

    pub fn get_params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.l1.w, &mut self.l1.b, &mut self.l2.w, &mut self.l2.b]
    }
}

pub struct SpherePpoTrainer {
    envs: Vec<RollingSphereEnv>,
    policy: PolicyNet,
    value: ValueNet,
    policy_optim: Adam,
    value_optim: Adam,
    gamma: f32,
    clip: f32,
    t_max: usize,
    n_epochs: usize,
    obs: Vec<Vec<f32>>,
}

impl SpherePpoTrainer {
    pub fn new(seed: u64) -> Self {
        fastrand::seed(seed);
        let mut policy = PolicyNet::new(1, 32, 1);
        let mut value = ValueNet::new(1, 32, 1);
        let policy_optim = Adam::new(&policy.get_params().iter().map(|t| &**t).collect::<Vec<&Tensor>>());
        let value_optim = Adam::new(&value.get_params().iter().map(|t| &**t).collect::<Vec<&Tensor>>());
        let obs = (0..8).map(|_| vec![0.0]).collect();
        Self {
            envs: (0..8).map(|_| RollingSphereEnv::new()).collect(),
            policy,
            value,
            policy_optim,
            value_optim,
            gamma: 0.99,
            clip: 0.2,
            t_max: 64,
            n_epochs: 3,
            obs
        }
    }

    pub fn step(&mut self) -> f32 {
        let mut all_obs = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_log_probs = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_dones = Vec::new();
        let mut all_values = Vec::new();
        let mut total_rewards = vec![0.0; self.envs.len()];

        for _ in 0..self.t_max {
            all_obs.push(self.obs.clone());
            let obs_tensor = Tensor::from_vec(vec![self.envs.len(), 1], self.obs.iter().flatten().copied().collect());
            let action_tensor = self.policy.forward(&obs_tensor, &mut Tape::new());
            let value_tensor = self.value.forward(&obs_tensor, &mut Tape::new());
            
            let actions = action_tensor.data.clone();
            all_actions.push(actions.clone());
            all_values.push(value_tensor.data.clone());
            all_log_probs.push(actions.iter().map(|a| -0.5 * a.powi(2)).collect::<Vec<_>>());

            let mut next_obs = Vec::new();
            let mut rewards = Vec::new();
            let mut dones = Vec::new();

            for (i, env) in self.envs.iter_mut().enumerate() {
                let (ob, reward, done) = env.step(actions[i]);
                next_obs.push(ob);
                rewards.push(reward);
                dones.push(done);
                total_rewards[i] += reward;
                if done {
                    self.obs[i] = env.reset();
                } else {
                    self.obs[i] = next_obs[i].clone();
                }
            }
            all_rewards.push(rewards);
            all_dones.push(dones);
        }

        let mut advantages = vec![vec![0.0; self.envs.len()]; self.t_max];
        let mut returns = vec![vec![0.0; self.envs.len()]; self.t_max];
        let mut last_advantage = vec![0.0; self.envs.len()];

        for t in (0..self.t_max).rev() {
            for i in 0..self.envs.len() {
                let next_value = if t == self.t_max - 1 { 0.0 } else { all_values[t + 1][i] };
                let delta = all_rewards[t][i] + self.gamma * next_value * (1.0 - all_dones[t][i] as i32 as f32) - all_values[t][i];
                advantages[t][i] = delta + self.gamma * 0.95 * last_advantage[i] * (1.0 - all_dones[t][i] as i32 as f32);
                last_advantage[i] = advantages[t][i];
                returns[t][i] = advantages[t][i] + all_values[t][i];
            }
        }
        
        let obs_flat: Vec<f32> = all_obs.into_iter().flatten().map(|v| v[0]).collect();
        let actions_flat: Vec<f32> = all_actions.into_iter().flatten().collect();
        let log_probs_flat: Vec<f32> = all_log_probs.into_iter().flatten().collect();
        let advantages_flat: Vec<f32> = advantages.into_iter().flatten().collect();
        let returns_flat: Vec<f32> = returns.into_iter().flatten().collect();

        let advantages_mean = advantages_flat.iter().sum::<f32>() / advantages_flat.len() as f32;
        let advantages_std = (advantages_flat.iter().map(|x| (x - advantages_mean).powi(2)).sum::<f32>() / advantages_flat.len() as f32).sqrt();
        let advantages_norm: Vec<f32> = advantages_flat.iter().map(|x| (x - advantages_mean) / (advantages_std + 1e-8)).collect();
        
        for _ in 0..self.n_epochs {
            let mut tape = Tape::new();
            let mut tensors = HashMap::new();
            let mut params = self.policy.get_params();
            params.extend(self.value.get_params());
            for p in params.iter_mut() {
                *p = p.clone().with_grad();
                tensors.insert(p.id, (*p).clone());
            }

            let obs_tensor = Tensor::from_vec(vec![obs_flat.len(), 1], obs_flat.clone());
            let action_tensor = self.policy.forward(&obs_tensor, &mut tape);
            let value_tensor = self.value.forward(&obs_tensor, &mut tape);
            tensors.insert(action_tensor.id, action_tensor.clone());
            tensors.insert(value_tensor.id, value_tensor.clone());
            
            let log_probs = action_tensor.sub(&Tensor::from_vec(vec![actions_flat.len()], actions_flat.clone()), &mut tape).pow(2.0, &mut tape).mul_scalar(-0.5, &mut tape);
            let advantages_tensor = Tensor::from_vec(vec![advantages_norm.len()], advantages_norm.clone());
            let returns_tensor = Tensor::from_vec(vec![returns_flat.len()], returns_flat.clone());
            let old_log_probs_tensor = Tensor::from_vec(vec![log_probs_flat.len()], log_probs_flat.clone());

            let ratio = log_probs.sub(&old_log_probs_tensor, &mut tape).exp(&mut tape);
            
            let policy_loss1 = ratio.mul(&advantages_tensor, &mut tape);
            let policy_loss2 = ratio.clone().clamp(1.0 - self.clip, 1.0 + self.clip, &mut tape).mul(&advantages_tensor, &mut tape);
            let policy_loss = policy_loss1.min(&policy_loss2, &mut tape).reduce_mean(&mut tape).mul_scalar(-1.0, &mut tape);

            let value_loss = value_tensor.sub(&returns_tensor, &mut tape).pow(2.0, &mut tape).reduce_mean(&mut tape);
            
            let loss = policy_loss.add(&value_loss, &mut tape);
            tensors.insert(loss.id, loss.clone());
            tape.backward(&loss, &mut tensors).unwrap();

            self.policy_optim.step(&mut self.policy.get_params());
            self.value_optim.step(&mut self.value.get_params());
        }

        total_rewards.iter().sum::<f32>() / total_rewards.len() as f32
    }
} 