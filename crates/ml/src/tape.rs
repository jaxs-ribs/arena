use crate::graph::{Node, EOp};
use crate::Tensor;
use anyhow::Result;
use crate::recorder::Recorder;
use std::collections::HashMap;

pub struct Tape {
    nodes: Vec<Node>,
}

impl Recorder for Tape {
    fn record(&mut self, node: Node) {
        self.nodes.push(node);
    }
    fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn backward(&mut self, loss: &Tensor, tensors: &mut HashMap<usize, Tensor>) -> Result<()> {
        let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
        grads.insert(loss.id, vec![1.0; loss.data.len()]);

        for node in self.nodes.iter().rev() {
            let out_grad = grads.get(&node.out).unwrap().clone();
            let a = tensors.get(&node.a).unwrap();
            let b = tensors.get(&node.b).unwrap();

            match node.op {
                EOp::Add => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                        *g += og;
                    }

                    let b_grad = grads.entry(node.b).or_insert_with(|| vec![0.0; b.data.len()]);
                    for (g, og) in b_grad.iter_mut().zip(out_grad.iter()) {
                        *g += og;
                    }
                }
                EOp::Mul => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    for (g, (d, og)) in a_grad.iter_mut().zip(b.data.iter().zip(out_grad.iter())) {
                        *g += d * og;
                    }

                    let b_grad = grads.entry(node.b).or_insert_with(|| vec![0.0; b.data.len()]);
                    for (g, (d, og)) in b_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        *g += d * og;
                    }
                }
                EOp::ReduceSum => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    for g in a_grad.iter_mut() {
                        *g += out_grad[0];
                    }
                }
                EOp::MatMul => {
                    // g_W += g_out * x^T
                    let w_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    for i in 0..a.shape[0] { // out_dim
                        for j in 0..a.shape[1] { // in_dim
                            w_grad[i * a.shape[1] + j] += out_grad[i] * b.data[j];
                        }
                    }

                    // g_x += W^T * g_out
                    let x_grad = grads.entry(node.b).or_insert_with(|| vec![0.0; b.data.len()]);
                    for i in 0..a.shape[0] { // out_dim
                        for j in 0..a.shape[1] { // in_dim
                            x_grad[j] += a.data[i * a.shape[1] + j] * out_grad[i];
                        }
                    }
                }
                EOp::Tanh => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let out = tensors.get(&node.out).unwrap();
                    for (g, (d, og)) in a_grad.iter_mut().zip(out.data.iter().zip(out_grad.iter())) {
                        *g += (1.0 - d.powi(2)) * og;
                    }
                }
                EOp::Sub => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                        *g += og;
                    }
                    let b_grad = grads.entry(node.b).or_insert_with(|| vec![0.0; b.data.len()]);
                    for (g, og) in b_grad.iter_mut().zip(out_grad.iter()) {
                        *g -= og;
                    }
                }
                EOp::Pow => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let exp = tensors.get(&node.b).unwrap().data[0];
                    for (g, (d, og)) in a_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        *g += exp * d.powf(exp - 1.0) * og;
                    }
                }
                EOp::MulScalar => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let scalar = tensors.get(&node.b).unwrap().data[0];
                    for (g, og) in a_grad.iter_mut().zip(out_grad.iter()) {
                        *g += scalar * og;
                    }
                }
                EOp::Clamp => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let out = tensors.get(&node.out).unwrap();
                    let min = tensors.get(&node.b).unwrap().data[0];
                    // This is a hack, max is not stored. Assuming it's in a magic tensor ID
                    let max = tensors.get(&(node.b + 1)).unwrap().data[0];
                    for (g, (d, og)) in a_grad.iter_mut().zip(a.data.iter().zip(out_grad.iter())) {
                        if *d > min && *d < max {
                            *g += og;
                        }
                    }
                }
                EOp::Min => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let b_grad = grads.entry(node.b).or_insert_with(|| vec![0.0; b.data.len()]);
                    let out = tensors.get(&node.out).unwrap();
                    for i in 0..out.data.len() {
                        if a.data[i] < b.data[i] {
                            a_grad[i] += out_grad[i];
                        } else {
                            b_grad[i] += out_grad[i];
                        }
                    }
                }
                EOp::ReduceMean => {
                    let a_grad = grads.entry(node.a).or_insert_with(|| vec![0.0; a.data.len()]);
                    let n = a.data.len() as f32;
                    for g in a_grad.iter_mut() {
                        *g += out_grad[0] / n;
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