pub mod graph;
pub mod tape;
pub mod nn;
pub mod optim;
pub mod rl;
pub mod recorder;

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::graph::{Node, EOp};
use crate::recorder::Recorder;

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data:  Vec<f32>,              // host shadow
    pub gpu:   Option<compute::BufferView>,
    pub id:    usize,
    pub requires_grad: bool,
    pub grad:  Option<Vec<f32>>,
}

impl Tensor {
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape,
            data,
            gpu: None,
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::SeqCst),
            requires_grad: false,
            grad: None,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn with_grad(mut self) -> Self {
        if self.requires_grad {
            return self;
        }
        self.grad = Some(vec![0.0; self.data.len()]);
        self.requires_grad = true;
        self
    }

    pub fn add(&self, other: &Self, recorder: &mut impl Recorder) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(Node { op: EOp::Add, a: self.id, b: other.id, out: out.id });
        out
    }

    pub fn mul(&self, other: &Self, recorder: &mut impl Recorder) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(Node { op: EOp::Mul, a: self.id, b: other.id, out: out.id });
        out
    }

    pub fn reduce_sum(&self, recorder: &mut impl Recorder) -> Self {
        let data = vec![self.data.iter().sum()];
        let out = Tensor::from_vec(vec![1], data);
        recorder.record(Node { op: EOp::ReduceSum, a: self.id, b: 0, out: out.id });
        out
    }

    pub fn matmul(&self, other: &Self, recorder: &mut impl Recorder) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 1);
        assert_eq!(self.shape[1], other.shape[0]);
        let out_shape = vec![self.shape[0]];
        let mut out_data = vec![0.0; out_shape[0]];

        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                out_data[i] += self.data[i * self.shape[1] + j] * other.data[j];
            }
        }

        let out = Tensor::from_vec(out_shape, out_data);
        recorder.record(Node { op: EOp::MatMul, a: self.id, b: other.id, out: out.id });
        out
    }

    pub fn tanh(&self, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(Node { op: EOp::Tanh, a: self.id, b: 0, out: out.id });
        out
    }

    pub fn sub(&self, other: &Self, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().zip(other.data.iter()).map(|(a,b)| a - b).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(Node { op: EOp::Sub, a: self.id, b: other.id, out: out.id });
        out
    }

    pub fn pow(&self, exp: f32, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().map(|x| x.powf(exp)).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let exp_tensor = Tensor::from_vec(vec![1], vec![exp]); // Not elegant
        recorder.record(Node { op: EOp::Pow, a: self.id, b: exp_tensor.id, out: out.id });
        out
    }

    pub fn mul_scalar(&self, scalar: f32, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().map(|x| x * scalar).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let scalar_tensor = Tensor::from_vec(vec![1], vec![scalar]); // Not elegant
        recorder.record(Node { op: EOp::MulScalar, a: self.id, b: scalar_tensor.id, out: out.id });
        out
    }

    pub fn clamp(&self, min: f32, max: f32, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().map(|x| x.clamp(min, max)).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let min_tensor = Tensor::from_vec(vec![1], vec![min]);
        let max_tensor = Tensor::from_vec(vec![1], vec![max]);
        // This is a hack, we should probably have a better way to handle multiple arguments
        recorder.record(Node { op: EOp::Clamp, a: self.id, b: min_tensor.id, out: out.id });
        out
    }

    pub fn min(&self, other: &Self, recorder: &mut impl Recorder) -> Self {
        let data = self.data.iter().zip(other.data.iter()).map(|(a,b)| a.min(*b)).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(Node { op: EOp::Min, a: self.id, b: other.id, out: out.id });
        out
    }

    pub fn reduce_mean(&self, recorder: &mut impl Recorder) -> Self {
        let sum = self.data.iter().sum::<f32>();
        let mean = sum / self.data.len() as f32;
        let out = Tensor::from_vec(vec![1], vec![mean]);
        recorder.record(Node { op: EOp::ReduceMean, a: self.id, b: 0, out: out.id });
        out
    }
} 