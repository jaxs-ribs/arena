use std::sync::Arc;
use compute::{ComputeBackend, ComputeError};
use crate::recorder::Recorder;

#[derive(Clone, Copy, Debug)]
pub enum EOp { Add, Mul, ReduceSum, MatMul, Tanh, Clamp, Min, Sub, Pow, MulScalar, ReduceMean }
#[derive(Clone)]
pub struct Node { pub op:EOp, pub a:usize, pub b:usize, pub out:usize }
pub struct Graph { nodes:Vec<Node> }

impl Recorder for Graph {
    fn record(&mut self, node: Node) {
        self.nodes.push(node);
    }

    fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }
}

impl Graph {
    pub fn new() -> Self { Self { nodes: Vec::new() } }
    pub fn run(&self, _backend:&Arc<dyn ComputeBackend>) -> Result<(),ComputeError> { Ok(()) }
} 