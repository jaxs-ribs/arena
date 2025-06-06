pub use crate::graph;
use crate::recorder::Recorder;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// A fully connected neural network layer.
#[derive(Clone)]
pub struct Dense {
    /// The weight matrix for the layer.
    pub w: Tensor,
    /// The bias vector for the layer.
    pub b: Tensor,
    /// The number of input dimensions.
    pub in_dim: usize,
    /// The number of output dimensions.
    pub out_dim: usize,
}

impl Dense {
    /// Creates a new `Dense` layer with the given weights and biases.
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

    /// Performs the forward pass through the layer.
    pub fn forward(
        &self,
        x: &Tensor,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Tensor {
        let wx = self.w.matmul(x, recorder, tensors);
        let y = wx.add_broadcast(&self.b, recorder, tensors);
        y
    }

    /// A helper function for finite-difference gradient checking.
    ///
    /// This function is not intended for use in production code.
    pub fn fd_loss(&self, w: &Tensor, x: &Tensor) -> Tensor {
        let mut layer = self.clone();
        layer.w = w.clone();
        let mut tensors = HashMap::new();
        tensors.insert(layer.w.id, layer.w.clone());
        tensors.insert(x.id, x.clone());
        tensors.insert(layer.b.id, layer.b.clone());
        let y = layer.forward(x, &mut graph::Graph::new(), &mut tensors);
        y.reduce_sum(&mut graph::Graph::new(), &mut tensors)
    }
}
