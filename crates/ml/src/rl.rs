use crate::env::Env;
use crate::stick_balance::StickBalanceEnv;

/// A simple environment where the agent must learn to roll a sphere to the right.
pub struct RollingSphereEnv {
    pos_x: f32,
}

impl RollingSphereEnv {
    /// Creates a new `RollingSphereEnv`.
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

    fn obs_size(&self) -> usize {
        1
    }
    fn action_size(&self) -> usize {
        1
    }
}

use crate::{nn::{self, Dense}, optim::Adam, tape::Tape, tensor::Tensor};
use std::collections::HashMap;

/// Simple policy/value network used by [`SpherePpoTrainer`].
struct PolicyValueNet {
    l1: Dense,
    policy_head: Dense,
    value_head: Dense,
}

impl PolicyValueNet {
    fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            l1: Dense::random(in_dim, hidden_dim, 0),
            policy_head: Dense::random(hidden_dim, out_dim, 1),
            value_head: Dense::random(hidden_dim, 1, 2),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        recorder: &mut impl crate::recorder::Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> (Tensor, Tensor) {
        let x = self.l1.forward(x, recorder, tensors);
        let x = x.tanh(recorder, tensors);
        let policy = self.policy_head.forward(&x, recorder, tensors);
        let value = self.value_head.forward(&x, recorder, tensors);
        (policy, value)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.l1.w,
            &mut self.l1.b,
            &mut self.policy_head.w,
            &mut self.policy_head.b,
            &mut self.value_head.w,
            &mut self.value_head.b,
        ]
    }
}

/// Generic trainer for the Proximal Policy Optimization (PPO) algorithm.
pub struct PpoTrainer<E: Env> {
    envs: Vec<E>,
    net: PolicyValueNet,
    optimizer: Adam,
    gamma: f32,
    lambda: f32,
    clip: f32,
    t_max: usize,
    n_epochs: usize,
    obs: Vec<Vec<f32>>,
    obs_dim: usize,
    act_dim: usize,
}

impl<E: Env> PpoTrainer<E> {
    /// Creates a new `PpoTrainer` with the provided environment constructor.
    pub fn new_with(mut make_env: impl FnMut() -> E, seed: u64) -> Self {
        fastrand::seed(seed);
        let envs: Vec<_> = (0..8).map(|_| make_env()).collect();
        let obs_dim = envs[0].obs_size();
        let act_dim = envs[0].action_size();
        assert_eq!(act_dim, 1, "only single-dimensional actions supported");

        let mut net = PolicyValueNet::new(obs_dim, 32, act_dim);
        let params_tmp = net.params();
        let params: Vec<&Tensor> = params_tmp.iter().map(|p| &**p).collect();
        let optimizer = Adam::new(&params);
        let obs = vec![vec![0.0; obs_dim]; envs.len()];
        Self {
            envs,
            net,
            optimizer,
            gamma: 0.99,
            lambda: 0.95,
            clip: 0.2,
            t_max: 64,
            n_epochs: 4,
            obs,
            obs_dim,
            act_dim,
        }
    }

    /// Performs a single training step.
    pub fn step(&mut self) -> f32 {
        // storage of trajectories
        let mut all_obs = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_log_probs = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_dones = Vec::new();
        let mut all_values = Vec::new();
        let mut total_rewards = vec![0.0; self.envs.len()];

        for _ in 0..self.t_max {
            all_obs.push(self.obs.clone());

            let obs_tensor = Tensor::from_vec(
                vec![self.envs.len(), self.obs_dim],
                self.obs.iter().flatten().copied().collect(),
            );
            let mut tmp_tensors = HashMap::new();
            tmp_tensors.insert(obs_tensor.id, obs_tensor.clone());
            let (action_tensor, value_tensor) =
                self.net.forward(&obs_tensor, &mut nn::graph::Graph::new(), &mut tmp_tensors);

            let actions: Vec<f32> = action_tensor.data().to_vec();
            let values: Vec<f32> = value_tensor.data().to_vec();
            all_actions.push(actions.clone());
            all_values.push(values);
            all_log_probs.push(
                actions
                    .iter()
                    .map(|a| -0.5 * a.powi(2) - 0.5 * (2.0 * std::f32::consts::PI).ln())
                    .collect::<Vec<_>>(),
            );

            let mut rewards = Vec::new();
            let mut dones = Vec::new();
            let mut next_obs = Vec::new();
            for (i, env) in self.envs.iter_mut().enumerate() {
                let (nobs, r, d) = env.step(actions[i]);
                next_obs.push(nobs);
                rewards.push(r);
                dones.push(d);
                total_rewards[i] += r;
                if d {
                    self.obs[i] = env.reset();
                } else {
                    self.obs[i] = next_obs[i].clone();
                }
            }
            all_rewards.push(rewards);
            all_dones.push(dones);
        }

        // compute value for last observation
        let obs_tensor = Tensor::from_vec(
            vec![self.envs.len(), self.obs_dim],
            self.obs.iter().flatten().copied().collect(),
        );
        let mut tmp = HashMap::new();
        tmp.insert(obs_tensor.id, obs_tensor.clone());
        let (_, last_values_tensor) =
            self.net.forward(&obs_tensor, &mut nn::graph::Graph::new(), &mut tmp);
        let last_values = last_values_tensor.data().to_vec();

        let mut advantages = vec![vec![0.0; self.envs.len()]; self.t_max];
        let mut returns = vec![vec![0.0; self.envs.len()]; self.t_max];
        let mut last_advantage = vec![0.0; self.envs.len()];

        for t in (0..self.t_max).rev() {
            for i in 0..self.envs.len() {
                let next_value = if t == self.t_max - 1 {
                    last_values[i]
                } else {
                    all_values[t + 1][i]
                };
                let next_done = if t == self.t_max - 1 { false } else { all_dones[t + 1][i] };
                let delta = all_rewards[t][i]
                    + self.gamma * next_value * (1.0 - next_done as i32 as f32)
                    - all_values[t][i];
                advantages[t][i] = delta
                    + self.gamma * self.lambda * last_advantage[i] * (1.0 - next_done as i32 as f32);
                last_advantage[i] = advantages[t][i];
                returns[t][i] = advantages[t][i] + all_values[t][i];
            }
        }

        let obs_flat: Vec<f32> = all_obs.into_iter().flatten().flatten().collect();
        let actions_flat: Vec<f32> = all_actions.into_iter().flatten().collect();
        let log_probs_flat: Vec<f32> = all_log_probs.into_iter().flatten().collect();
        let advantages_flat: Vec<f32> = advantages.iter().flatten().copied().collect();
        let returns_flat: Vec<f32> = returns.iter().flatten().copied().collect();

        let mean_adv: f32 = advantages_flat.iter().sum::<f32>() / advantages_flat.len() as f32;
        let std_adv: f32 =
            (advantages_flat.iter().map(|x| (x - mean_adv).powi(2)).sum::<f32>() / advantages_flat.len() as f32).sqrt();
        let advantages_norm: Vec<f32> =
            advantages_flat.iter().map(|x| (x - mean_adv) / (std_adv + 1e-8)).collect();

        for _ in 0..self.n_epochs {
            let mut tape = Tape::new();
            let mut tensors = HashMap::new();

            for p in self.net.params().iter_mut() {
                p.set_requires_grad();
                tensors.insert(p.id, (*p).clone());
            }

            let obs_tensor = Tensor::from_vec(
                vec![obs_flat.len() / self.obs_dim, self.obs_dim],
                obs_flat.clone(),
            );
            tensors.insert(obs_tensor.id, obs_tensor.clone());
            let (action_tensor, value_tensor) = self.net.forward(&obs_tensor, &mut tape, &mut tensors);

            let actions_tensor = Tensor::from_vec(
                vec![actions_flat.len() / self.act_dim, self.act_dim],
                actions_flat.clone(),
            );
            tensors.insert(actions_tensor.id, actions_tensor.clone());
            let log_probs = action_tensor
                .sub(&actions_tensor, &mut tape, &mut tensors)
                .pow(2.0, &mut tape, &mut tensors)
                .mul_scalar(-0.5, &mut tape, &mut tensors);

            let advantages_tensor = Tensor::from_vec(vec![advantages_norm.len(), 1], advantages_norm.clone());
            tensors.insert(advantages_tensor.id, advantages_tensor.clone());
            let returns_tensor = Tensor::from_vec(vec![returns_flat.len(), 1], returns_flat.clone());
            tensors.insert(returns_tensor.id, returns_tensor.clone());
            let old_log_probs_tensor =
                Tensor::from_vec(vec![log_probs_flat.len(), 1], log_probs_flat.clone());
            tensors.insert(old_log_probs_tensor.id, old_log_probs_tensor.clone());

            let ratio = log_probs
                .sub(&old_log_probs_tensor, &mut tape, &mut tensors)
                .exp(&mut tape, &mut tensors);

            let policy_loss1 = ratio.mul(&advantages_tensor, &mut tape, &mut tensors);
            let policy_loss2 = ratio
                .clone()
                .clamp(1.0 - self.clip, 1.0 + self.clip, &mut tape, &mut tensors)
                .mul(&advantages_tensor, &mut tape, &mut tensors);
            let policy_loss = policy_loss1
                .min(&policy_loss2, &mut tape, &mut tensors)
                .reduce_mean(&mut tape, &mut tensors)
                .mul_scalar(-1.0, &mut tape, &mut tensors);

            let value_loss = value_tensor
                .sub(&returns_tensor, &mut tape, &mut tensors)
                .pow(2.0, &mut tape, &mut tensors)
                .reduce_mean(&mut tape, &mut tensors);

            let loss = policy_loss.add(&value_loss, &mut tape, &mut tensors);
            tensors.insert(loss.id, loss.clone());

            tape.backward(&loss, &mut tensors).unwrap();
            self.optimizer.step(&mut self.net.params());
        }

        total_rewards.iter().sum::<f32>() / total_rewards.len() as f32
    }

    /// Returns an action for the provided observation using the current policy.
    pub fn act(&self, obs: &[f32]) -> f32 {
        assert_eq!(obs.len(), self.obs_dim);
        let obs_tensor = Tensor::from_vec(vec![1, self.obs_dim], obs.to_vec());
        let mut tensors = HashMap::new();
        tensors.insert(obs_tensor.id, obs_tensor.clone());
        let (action_tensor, _value_tensor) =
            self.net
                .forward(&obs_tensor, &mut nn::graph::Graph::new(), &mut tensors);
        action_tensor.data()[0]
    }
}

pub type SpherePpoTrainer = PpoTrainer<RollingSphereEnv>;
pub type StickBalancePpoTrainer = PpoTrainer<StickBalanceEnv>;

impl PpoTrainer<RollingSphereEnv> {
    /// Convenience constructor for the rolling sphere task.
    pub fn new(seed: u64) -> Self {
        Self::new_with(|| RollingSphereEnv::new(), seed)
    }
}

impl PpoTrainer<StickBalanceEnv> {
    /// Convenience constructor for the stick balancing task.
    pub fn new(seed: u64) -> Self {
        Self::new_with(|| StickBalanceEnv::new(), seed)
    }
}
