use crate::graph::Node;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// A trait for recording operations in a computation graph.
pub trait Recorder {
    /// Records a single operation node.
    fn record(&mut self, node: Node, tensors: &mut HashMap<usize, Tensor>);
    /// Returns an immutable reference to the list of nodes recorded.
    fn nodes(&self) -> &Vec<Node>;
}
