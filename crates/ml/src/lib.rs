use compute::{BufferView, ComputeBackend, ComputeError, Kernel};
use std::sync::Arc;
use bytemuck;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub gpu: Option<BufferView>,
}

impl Tensor {
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { data, shape, gpu: None }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Add,
    Mul,
    Sub,
    Div,
    Where,
    Exp,
    Log,
    Tanh,
}

#[derive(Clone)]
pub struct OpCall { pub op: Op, pub a: Tensor, pub b: Tensor, pub out: Tensor }

pub struct Graph { pub calls: Vec<OpCall> }

impl Default for Graph { fn default() -> Self { Self { calls: Vec::new() } } }

impl Graph {
    pub fn add_call(&mut self, call: OpCall) { self.calls.push(call); }

    pub fn run<B: ComputeBackend + ?Sized>(&mut self, _backend: &B) -> Result<(), ComputeError> {
        Ok(())
    }
}

fn upload(t: &mut Tensor) -> BufferView {
    if let Some(v) = &t.gpu { return v.clone(); }
    let bytes: Arc<[u8]> = bytemuck::cast_slice(&t.data).to_vec().into();
    let view = BufferView::new(bytes, t.shape.clone(), std::mem::size_of::<f32>());
    t.gpu = Some(view.clone());
    view
}

fn make_buffer(len: usize) -> BufferView {
    let bytes: Arc<[u8]> = vec![0u8; len*4].into();
    BufferView::new(bytes, vec![len], 4)
}

pub struct Dense {
    pub w: Tensor,
    pub b: Tensor,
    in_dim: usize,
    out_dim: usize,
}

impl Dense {
    pub fn new(weights: Vec<f32>, bias: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        assert_eq!(weights.len(), in_dim * out_dim);
        assert_eq!(bias.len(), out_dim);
        Self {
            w: Tensor::from_vec(vec![out_dim, in_dim], weights),
            b: Tensor::from_vec(vec![out_dim], bias),
            in_dim,
            out_dim,
        }
    }

    pub fn forward(&self, x: &Tensor, _g: &mut Graph) -> Tensor {
        let dummy_data = vec![0.0f32; self.out_dim];
        Tensor::from_vec(vec![self.out_dim], dummy_data)
    }
}
