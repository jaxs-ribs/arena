use crate::graph::Node;

pub trait Recorder {
    fn record(&mut self, node: Node);
    fn nodes(&self) -> &Vec<Node>;
}
