use crate::Tensor;
use crate::recorder::Recorder;
use crate::graph;

#[derive(Clone)]
pub struct Dense {
    pub w: Tensor,
    pub b: Tensor,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Dense {
    pub fn new(weights: Vec<f32>, bias: Vec<f32>, in_d: usize, out_d: usize) -> Self {
        assert_eq!(weights.len(), in_d * out_d);
        assert_eq!(bias.len(), out_d);
        Self {
            w: Tensor::from_vec(vec![out_d, in_d], weights),
            b: Tensor::from_vec(vec![out_d], bias),
            in_dim: in_d,
            out_dim: out_d,
        }
    }

    pub fn random(in_d: usize, out_d: usize, _seed: u64) -> Self {
        // Glorot initialization
        let limit = (6.0 / (in_d + out_d) as f32).sqrt();
        let weights = (0..in_d * out_d)
            .map(|_| fastrand::f32() * 2.0 * limit - limit)
            .collect();
        let bias = vec![0.0; out_d];
        Self::new(weights, bias, in_d, out_d)
    }

    pub fn forward(&self, x: &Tensor, recorder: &mut impl Recorder) -> (Tensor, Tensor) {
        let wx = self.w.matmul(x, recorder);
        let y = wx.add(&self.b, recorder);
        (y, wx)
    }

    pub fn fd_loss(&self, w: &Tensor, x: &Tensor) -> Tensor {
        let mut layer = self.clone();
        layer.w = w.clone();
        let (y, _) = layer.forward(x, &mut graph::Graph::new());
        y.reduce_sum(&mut graph::Graph::new())
    }
} 