/// A trait for reinforcement learning environments.
pub trait Env {
    /// Steps the environment forward by one time-step.
    fn step(&mut self, action: f32) -> (Vec<f32>, f32, bool);
    /// Resets the environment to an initial state.
    fn reset(&mut self) -> Vec<f32>;
    /// Returns the size of the observation space.
    fn obs_size(&self) -> usize;
    /// Returns the size of the action space.
    fn action_size(&self) -> usize;
}

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

/// A trainer for the Proximal Policy Optimization (PPO) algorithm.
pub struct SpherePpoTrainer {}

impl SpherePpoTrainer {
    /// Creates a new `SpherePpoTrainer`.
    pub fn new(_seed: u64) -> Self {
        unimplemented!()
    }

    /// Performs a single training step.
    pub fn step(&mut self) -> f32 {
        unimplemented!()
    }
}
