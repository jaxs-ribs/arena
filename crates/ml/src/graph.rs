use crate::recorder::Recorder;
use crate::tensor::Tensor;
use compute::{ComputeBackend, ComputeError};
use std::collections::HashMap;
use std::sync::Arc;

/// An enumeration of the possible operations in a computation graph.
#[derive(Clone, Copy, Debug)]
pub enum EOp {
    Add,
    Mul,
    ReduceSum,
    MatMul,
    Tanh,
    Sub,
    Pow,
    MulScalar,
    Clamp,
    Min,
    ReduceMean,
    Exp,
    AddBroadcast,
}
/// A node in the computation graph, representing a single operation.
#[derive(Clone)]
pub struct Node {
    pub op: EOp,
    pub a: usize,
    pub b: usize,
    pub out: usize,
}
/// A computation graph that records operations but does not compute gradients.
pub struct Graph {
    nodes: Vec<Node>,
}

impl Recorder for Graph {
    fn record(&mut self, node: Node, _tensors: &mut HashMap<usize, Tensor>) {
        self.nodes.push(node);
    }

    fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }
}

impl Graph {
    /// Creates a new, empty computation graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }
    /// Executes the computation graph on a given backend.
    /// Note: This is a placeholder and does not yet support GPU execution.
    pub fn run(&self, _backend: &Arc<dyn ComputeBackend>) -> Result<(), ComputeError> {
        Ok(())
    }
}
