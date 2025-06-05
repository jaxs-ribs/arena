use ml::{Dense, Tensor, Graph};

pub struct PpoAgent {
    policy: Dense,
    lr: f32,
}

impl PpoAgent {
    #[must_use]
    pub fn new(lr: f32) -> Self {
        // single-input linear policy
        let weights = vec![0.0];
        let bias = vec![0.0];
        Self { policy: Dense::new(weights, bias, 1, 1), lr }
    }

    #[must_use]
    pub fn act(&self, obs: &Tensor) -> f32 {
        let mut g = Graph::default();
        let out = self.policy.forward(obs, &mut g);
        out.data[0]
    }

    pub fn update(&mut self, obs: &Tensor, action: f32, advantage: f32) {
        // gradient of linear layer w.r.t weights is obs * advantage
        let grad = obs.data[0] * advantage;
        self.policy.w.data[0] += self.lr * grad;
        // bias gradient
        self.policy.b.data[0] += self.lr * advantage;
    }
}
