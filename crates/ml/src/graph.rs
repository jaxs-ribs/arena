use crate::recorder::Recorder;
use crate::tensor::Tensor;
use compute::ComputeError;
use std::collections::HashMap;

/// An enumeration of the possible operations in a computation graph.
#[derive(Clone, Copy, Debug)]
pub enum EOp {
    Add,
    Mul,
    Div,
    ReduceSum,
    MatMul,
    Tanh,
    Relu,
    Sigmoid,
    Log,
    Sqrt,
    Rsqrt,
    Sub,
    Pow,
    MulScalar,
    Clamp,
    Min,
    Max,
    ReduceMax,
    ReduceMean,
    Exp,
    AddBroadcast,
    Neg,
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
    /// Executes the computation graph using the [`compute`] backend.
    ///
    /// The tensors for each recorded node must be provided via `tensors`.
    /// Results are written back into the output tensors contained in `tensors`.
    pub fn run(&self, tensors: &mut HashMap<usize, Tensor>) -> Result<(), ComputeError> {
        use compute::{BufferView, Kernel};

        let backend = compute::default_backend();

        for node in &self.nodes {
            let kernel = match node.op {
                EOp::Add => Kernel::Add,
                EOp::Mul => Kernel::Mul,
                EOp::Div => Kernel::Div,
                EOp::ReduceSum => Kernel::ReduceSum,
                // Unsupported ops map to available kernels when possible
                EOp::MatMul => Kernel::MatMul,
                EOp::Tanh => Kernel::Tanh,
                EOp::Relu => Kernel::Relu,
                EOp::Sigmoid => Kernel::Sigmoid,
                EOp::Log => Kernel::Log,
                EOp::Sqrt => Kernel::Sqrt,
                EOp::Rsqrt => Kernel::Rsqrt,
                EOp::Sub => Kernel::Sub,
                EOp::Clamp => Kernel::Clamp,
                EOp::Min => Kernel::Min,
                EOp::Max => Kernel::Max,
                EOp::ReduceMax => Kernel::ReduceMax,
                EOp::ReduceMean => Kernel::ReduceMean,
                EOp::Exp => Kernel::Exp,
                EOp::AddBroadcast => Kernel::AddBroadcast,
                EOp::Neg => Kernel::Neg,
                _ => return Err(ComputeError::BackendUnavailable),
            };

            let a = tensors.get(&node.a).expect("tensor a missing");
            let a_view = BufferView::new(
                bytemuck::cast_slice(&a.data).to_vec().into(),
                a.shape.clone(),
                std::mem::size_of::<f32>(),
            );

            let mut binds: Vec<BufferView> = Vec::new();
            binds.push(a_view);

            match node.op {
                EOp::Add
                | EOp::Mul
                | EOp::Sub
                | EOp::Div
                | EOp::Min
                | EOp::Max => {
                    let b = tensors.get(&node.b).expect("tensor b missing");
                    let b_view = BufferView::new(
                        bytemuck::cast_slice(&b.data).to_vec().into(),
                        b.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(b_view);

                    let out = tensors.get(&node.out).expect("output tensor missing");
                    let out_placeholder = BufferView::new(
                        vec![0u8; out.data.len() * std::mem::size_of::<f32>()].into(),
                        out.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(out_placeholder);

                    // simple config buffer
                    let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
                    binds.push(cfg);

                    let result = backend.dispatch(&kernel, &binds, [1, 1, 1])?;
                    let out_tensor = tensors.get_mut(&node.out).expect("output tensor missing");
                    out_tensor.data = bytemuck::cast_slice(&result[0]).to_vec();
                }
                EOp::AddBroadcast => {
                    let b = tensors.get(&node.b).expect("tensor b missing");
                    let b_view = BufferView::new(
                        bytemuck::cast_slice(&b.data).to_vec().into(),
                        b.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(b_view);

                    let out = tensors.get(&node.out).expect("output tensor missing");
                    let out_placeholder = BufferView::new(
                        vec![0u8; out.data.len() * std::mem::size_of::<f32>()].into(),
                        out.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(out_placeholder);

                    // simple config buffer
                    let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
                    binds.push(cfg);

                    let result = backend.dispatch(&kernel, &binds, [1, 1, 1])?;
                    let out_tensor = tensors.get_mut(&node.out).expect("output tensor missing");
                    out_tensor.data = bytemuck::cast_slice(&result[0]).to_vec();
                }
                EOp::ReduceSum | EOp::ReduceMean | EOp::ReduceMax => {
                    let out = tensors.get(&node.out).expect("output tensor missing");
                    let out_placeholder = BufferView::new(
                        vec![0u8; std::mem::size_of::<f32>()].into(),
                        vec![1],
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(out_placeholder);
                    let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
                    binds.push(cfg);

                    let result = backend.dispatch(&kernel, &binds, [1, 1, 1])?;
                    let out_tensor = tensors.get_mut(&node.out).expect("output tensor missing");
                    out_tensor.data = vec![*bytemuck::from_bytes(&result[0])];
                }
                EOp::MatMul => {
                    let b = tensors.get(&node.b).expect("tensor b missing");
                    let b_view = BufferView::new(
                        bytemuck::cast_slice(&b.data).to_vec().into(),
                        b.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(b_view);

                    let out = tensors.get(&node.out).expect("output tensor missing");
                    let out_placeholder = BufferView::new(
                        vec![0u8; out.data.len() * std::mem::size_of::<f32>()].into(),
                        out.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(out_placeholder);

                    #[repr(C)]
                    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
                    struct MatMulConfig {
                        m: u32,
                        k: u32,
                        n: u32,
                    }

                    let m = a.shape[0] as u32;
                    let k = a.shape[1] as u32;
                    let n = b.shape[0] as u32;
                    let cfg_struct = MatMulConfig { m, k, n };
                    let cfg = BufferView::new(
                        bytemuck::bytes_of(&cfg_struct).to_vec().into(),
                        vec![1],
                        std::mem::size_of::<MatMulConfig>(),
                    );
                    binds.push(cfg);

                    let result = backend.dispatch(&kernel, &binds, [1, 1, 1])?;
                    let out_tensor = tensors.get_mut(&node.out).expect("output tensor missing");
                    out_tensor.data = bytemuck::cast_slice(&result[0]).to_vec();
                }
                EOp::Tanh
                | EOp::Relu
                | EOp::Sigmoid
                | EOp::Log
                | EOp::Sqrt
                | EOp::Rsqrt
                | EOp::Exp
                | EOp::Clamp
                | EOp::Neg => {
                    let out = tensors.get(&node.out).expect("output tensor missing");
                    let out_placeholder = BufferView::new(
                        vec![0u8; out.data.len() * std::mem::size_of::<f32>()].into(),
                        out.shape.clone(),
                        std::mem::size_of::<f32>(),
                    );
                    binds.push(out_placeholder);
                    let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
                    binds.push(cfg);

                    let result = backend.dispatch(&kernel, &binds, [1, 1, 1])?;
                    let out_tensor = tensors.get_mut(&node.out).expect("output tensor missing");
                    out_tensor.data = bytemuck::cast_slice(&result[0]).to_vec();
                }
                _ => {}
            }
        }

        Ok(())
    }
}
