pub mod graph;
pub mod nn;
pub mod optim;
pub mod recorder;
pub mod rl;
pub mod tape;

use crate::graph::{EOp, Node};
use crate::recorder::Recorder;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

/// A multi-dimensional array for computation.
#[derive(Clone)]
pub struct Tensor {
    /// The dimensions of the tensor.
    pub shape: Vec<usize>,
    /// The data buffer for the tensor, stored on the host.
    pub data: Vec<f32>, // host shadow
    /// An optional handle to a GPU buffer.
    pub gpu: Option<compute::BufferView>,
    /// A unique identifier for the tensor.
    pub id: usize,
    /// Whether this tensor requires a gradient to be computed.
    pub requires_grad: bool,
    /// The gradient of this tensor, if computed.
    pub grad: Option<Vec<f32>>,
}

impl Tensor {
    /// Creates a new tensor from a vector of data and a shape.
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

    /// Sets the requires_grad flag to true and initializes the gradient to zero.
    pub fn set_requires_grad(&mut self) {
        if self.requires_grad {
            return;
        }
        self.grad = Some(vec![0.0; self.data.len()]);
        self.requires_grad = true;
    }

    /// Performs element-wise addition between two tensors.
    pub fn add(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Add,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Performs element-wise multiplication between two tensors.
    pub fn mul(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Mul,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Reduces a tensor to a single value by summing all its elements.
    pub fn reduce_sum(
        &self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        let data = vec![self.data.iter().sum()];
        let out = Tensor::from_vec(vec![1], data);
        recorder.record(
            Node {
                op: EOp::ReduceSum,
                a: self.id,
                b: 0,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Performs a matrix-vector multiplication.
    /// The first tensor (`self`) must be a 2D matrix, and the second (`other`) must be a 2D batch of vectors.
    pub fn matmul(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape.len(), 2, "matmul self must be 2D");
        assert_eq!(other.shape.len(), 2, "matmul other must be 2D");
        let out_dim = self.shape[0];
        let in_dim = self.shape[1];
        assert_eq!(in_dim, other.shape[1]);
        let batch_size = other.shape[0];

        let out_shape = vec![batch_size, out_dim];
        let mut out_data = vec![0.0; batch_size * out_dim];

        for b in 0..batch_size {
            for i in 0..out_dim {
                for j in 0..in_dim {
                    out_data[b * out_dim + i] +=
                        self.data[i * in_dim + j] * other.data[b * in_dim + j];
                }
            }
        }

        let out = Tensor::from_vec(out_shape, out_data);
        recorder.record(
            Node {
                op: EOp::MatMul,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Adds a vector to each row of a matrix.
    pub fn add_broadcast(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 1);
        let batch_size = self.shape[0];
        let dim = self.shape[1];
        assert_eq!(dim, other.shape[0]);

        let mut out_data = self.data.clone();
        for b in 0..batch_size {
            for i in 0..dim {
                out_data[b * dim + i] += other.data[i];
            }
        }
        let out = Tensor::from_vec(self.shape.clone(), out_data);
        recorder.record(
            Node {
                op: EOp::AddBroadcast,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Applies the hyperbolic tangent function to each element of the tensor.
    pub fn tanh(&self, recorder: &mut impl Recorder, tensors: &mut HashMap<usize, Tensor>) -> Self {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Tanh,
                a: self.id,
                b: 0,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Performs element-wise subtraction between two tensors.
    pub fn sub(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Sub,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Raises each element of the tensor to a power.
    pub fn pow(
        &self,
        exp: f32,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        let data = self.data.iter().map(|x| x.powf(exp)).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let exp_tensor = Tensor::from_vec(vec![1], vec![exp]);
        tensors.insert(exp_tensor.id, exp_tensor.clone());
        recorder.record(
            Node {
                op: EOp::Pow,
                a: self.id,
                b: exp_tensor.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Multiplies each element of the tensor by a scalar.
    pub fn mul_scalar(
        &self,
        scalar: f32,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        let data = self.data.iter().map(|x| x * scalar).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let scalar_tensor = Tensor::from_vec(vec![1], vec![scalar]);
        tensors.insert(scalar_tensor.id, scalar_tensor.clone());
        recorder.record(
            Node {
                op: EOp::MulScalar,
                a: self.id,
                b: scalar_tensor.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Clamps each element of the tensor to a given range.
    pub fn clamp(
        &self,
        min: f32,
        max: f32,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        let data = self.data.iter().map(|x| x.clamp(min, max)).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        let min_max_tensor = Tensor::from_vec(vec![2], vec![min, max]);
        tensors.insert(min_max_tensor.id, min_max_tensor.clone());
        recorder.record(
            Node {
                op: EOp::Clamp,
                a: self.id,
                b: min_max_tensor.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Computes the element-wise minimum of two tensors.
    pub fn min(
        &self,
        other: &Self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Min,
                a: self.id,
                b: other.id,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Reduces a tensor to a single value by taking the mean of all its elements.
    pub fn reduce_mean(
        &self,
        recorder: &mut impl Recorder,
        tensors: &mut HashMap<usize, Tensor>,
    ) -> Self {
        let sum = self.data.iter().sum::<f32>();
        let mean = sum / self.data.len() as f32;
        let out = Tensor::from_vec(vec![1], vec![mean]);
        recorder.record(
            Node {
                op: EOp::ReduceMean,
                a: self.id,
                b: 0,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }

    /// Applies the exponential function to each element of the tensor.
    pub fn exp(&self, recorder: &mut impl Recorder, tensors: &mut HashMap<usize, Tensor>) -> Self {
        let data = self.data.iter().map(|x| x.exp()).collect();
        let out = Tensor::from_vec(self.shape.clone(), data);
        recorder.record(
            Node {
                op: EOp::Exp,
                a: self.id,
                b: 0,
                out: out.id,
            },
            tensors,
        );
        tensors.insert(out.id, out.clone());
        out
    }
}
