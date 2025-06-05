use crate::Tensor;

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(params: &[&Tensor]) -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: params.iter().map(|p| vec![0.0; p.data.len()]).collect(),
            v: params.iter().map(|p| vec![0.0; p.data.len()]).collect(),
        }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor]) {
        self.t += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt() / (1.0 - self.beta1.powi(self.t as i32)).sqrt();

        for (i, p) in params.iter_mut().enumerate() {
            let grad = p.grad.as_ref().unwrap();
            for j in 0..p.data.len() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad[j].powi(2);
                p.data[j] -= lr_t * self.m[i][j] / (self.v[i][j].sqrt() + self.eps);
            }
        }
    }
} 