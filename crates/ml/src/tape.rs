use crate::graph::{EOp, Node};
use crate::recorder::Recorder;
use crate::Tensor;
use anyhow::Result;
use std::collections::HashMap;

/// A tape that records operations for automatic differentiation.
pub struct Tape {
    nodes: Vec<Node>,
}

impl Recorder for Tape {
    fn record(&mut self, node: Node, _tensors: &mut HashMap<usize, Tensor>) {
        self.nodes.push(node);
        // The tape doesn't own the tensors, but it needs to know about them for the backward pass.
        // The caller is responsible for creating the output tensor and passing it to record.
        // This is a bit of a hack.
    }
    fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }
}

impl Tape {
    /// Creates a new, empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Computes the gradients of the tensors on the tape with respect to a loss tensor.
    ///
    /// The gradients are computed by traversing the recorded operations in reverse order.
    pub fn backward(&mut self, loss: &Tensor, tensors: &mut HashMap<usize, Tensor>) -> Result<()> {
        let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
        grads.insert(loss.id, vec![1.0; loss.data.len()]);

        for node in self.nodes.iter().rev() {
            let out_grad = grads.get(&node.out).unwrap().clone();

            match node.op {
                EOp::Add => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                        *g += og;
                    }

                    let b_grad = grads
                        .entry(node.b)
                        .or_insert_with(|| vec![0.0; b.data.len()]);
                    for (g, og) in b_grad.iter_mut().zip(out_grad.iter()) {
                        *g += og;
                    }
                }
                EOp::Mul => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    for (g, (d, og)) in a_grad.iter_mut().zip(b.data.iter().zip(out_grad.iter())) {
                        *g += d * og;
                    }

                    let b_grad = grads
                        .entry(node.b)
                        .or_insert_with(|| vec![0.0; b.data.len()]);
                    for (g, (d, og)) in b_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        *g += d * og;
                    }
                }
                EOp::ReduceSum => {
                    let a = tensors.get(&node.a).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    for g in a_grad.iter_mut() {
                        *g += out_grad[0];
                    }
                }
                EOp::MatMul => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let out_dim = a.shape[0];
                    let in_dim = a.shape[1];
                    let batch_size = b.shape[0];

                    {
                        let w_grad = grads
                            .entry(node.a)
                            .or_insert_with(|| vec![0.0; a.data.len()]);
                        for i in 0..out_dim {
                            for j in 0..in_dim {
                                for k in 0..batch_size {
                                    w_grad[i * in_dim + j] +=
                                        out_grad[k * out_dim + i] * b.data[k * in_dim + j];
                                }
                            }
                        }
                    }

                    {
                        let x_grad = grads
                            .entry(node.b)
                            .or_insert_with(|| vec![0.0; b.data.len()]);
                        for k in 0..batch_size {
                            for j in 0..in_dim {
                                for i in 0..out_dim {
                                    x_grad[k * in_dim + j] +=
                                        out_grad[k * out_dim + i] * a.data[i * in_dim + j];
                                }
                            }
                        }
                    }
                }
                EOp::AddBroadcast => {
                    let a = tensors.get(&node.a).unwrap();
                    {
                        let a_grad = grads
                            .entry(node.a)
                            .or_insert_with(|| vec![0.0; a.data.len()]);
                        for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                            *g += og;
                        }
                    }
                    {
                        let b = tensors.get(&node.b).unwrap();
                        let b_grad = grads
                            .entry(node.b)
                            .or_insert_with(|| vec![0.0; b.data.len()]);
                        let batch_size = a.shape[0];
                        let dim = a.shape[1];
                        for b_idx in 0..batch_size {
                            for i in 0..dim {
                                b_grad[i] += out_grad[b_idx * dim + i];
                            }
                        }
                    }
                }
                EOp::Tanh => {
                    let a = tensors.get(&node.a).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let out = tensors.get(&node.out).unwrap();
                    for (g, (d, og)) in a_grad.iter_mut().zip(out.data.iter().zip(out_grad.iter()))
                    {
                        *g += (1.0 - d.powi(2)) * og;
                    }
                }
                EOp::Sub => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    {
                        let a_grad = grads
                            .entry(node.a)
                            .or_insert_with(|| vec![0.0; a.data.len()]);
                        for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                            *g += og;
                        }
                    }
                    {
                        let b_grad = grads
                            .entry(node.b)
                            .or_insert_with(|| vec![0.0; b.data.len()]);
                        for (g, og) in b_grad.iter_mut().zip(out_grad.iter()) {
                            *g -= og;
                        }
                    }
                }
                EOp::Pow => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let exp = b.data[0];
                    for (g, (d, og)) in a_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        *g += exp * d.powf(exp - 1.0) * og;
                    }
                }
                EOp::MulScalar => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let scalar = b.data[0];
                    for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                        *g += scalar * og;
                    }
                }
                EOp::Clamp => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let min_max = &b.data;
                    let min = min_max[0];
                    let max = min_max[1];
                    for (g, (d, og)) in a_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        if *d > min && *d < max {
                            *g += og;
                        }
                    }
                }
                EOp::Min => {
                    let a = tensors.get(&node.a).unwrap();
                    let b = tensors.get(&node.b).unwrap();
                    {
                        let a_grad = grads
                            .entry(node.a)
                            .or_insert_with(|| vec![0.0; a.data.len()]);
                        for i in 0..a.data.len() {
                            if a.data[i] < b.data[i] {
                                a_grad[i] += out_grad[i];
                            }
                        }
                    }
                    {
                        let b_grad = grads
                            .entry(node.b)
                            .or_insert_with(|| vec![0.0; b.data.len()]);
                        for i in 0..b.data.len() {
                            if b.data[i] <= a.data[i] {
                                b_grad[i] += out_grad[i];
                            }
                        }
                    }
                }
                EOp::ReduceMean => {
                    let a = tensors.get(&node.a).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let n = a.data.len() as f32;
                    for g in a_grad.iter_mut() {
                        *g += out_grad[0] / n;
                    }
                }
                EOp::Exp => {
                    let a = tensors.get(&node.a).unwrap();
                    let a_grad = grads
                        .entry(node.a)
                        .or_insert_with(|| vec![0.0; a.data.len()]);
                    let out = tensors.get(&node.out).unwrap();
                    for (g, (d, og)) in a_grad.iter_mut().zip(out.data.iter().zip(out_grad.iter()))
                    {
                        *g += d * og;
                    }
                }
            }
        }

        for (id, grad) in grads {
            if let Some(tensor) = tensors.get_mut(&id) {
                if tensor.requires_grad {
                    tensor.grad = Some(grad);
                }
            }
        }

        Ok(())
    }
}
