use ml::{Dense, Tensor, Graph};

pub struct PpoAgent {
    // policy: Dense, // Dense::forward is now stubbed
    _lr: f32, // Renamed to avoid unused warning if policy is commented
}

impl PpoAgent {
    #[must_use]
    pub fn new(lr: f32) -> Self {
        // single-input linear policy
        // let weights = vec![0.0];
        // let bias = vec![0.0];
        // Self { policy: Dense::new(weights, bias, 1, 1), lr }
        Self { _lr: lr }
    }

    #[must_use]
    pub fn act(&self, _obs: &Tensor) -> f32 {
        /* TODO: Reimplement with functional ml::Dense
        let mut g = Graph::default();
        let out = self.policy.forward(obs, &mut g);
        out.data[0]
        */
        0.0 // Placeholder action
    }

    pub fn update(&mut self, _obs: &Tensor, _action: f32, _advantage: f32) {
        /* TODO: Reimplement with functional ml::Dense and gradient logic
        // gradient of linear layer w.r.t weights is obs * advantage
        let grad = obs.data[0] * advantage;
        self.policy.w.data[0] += self.lr * grad;
        // bias gradient
        self.policy.b.data[0] += self.lr * advantage;
        */
    }
}
