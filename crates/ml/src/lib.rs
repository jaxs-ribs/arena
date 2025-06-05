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

#[derive(Clone, Copy)]
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

    pub fn run(&mut self, backend: &impl ComputeBackend) -> Result<(), ComputeError> {
        for call in &mut self.calls {
            let len = call.out.len();
            let a_view = upload(&mut call.a);
            let b_view = upload(&mut call.b);
            let out_view = make_buffer(len);
            call.out.gpu = Some(out_view.clone());
            let cfg = match call.op {
                Op::Add => 0u32,
                Op::Mul => 1,
                Op::Sub => 2,
                Op::Div => 3,
                Op::Where => 4,
                Op::Exp => 5,
                Op::Log => 6,
                Op::Tanh => 7,
            };
            let cfg_bytes = cfg.to_le_bytes();
            let cfg_view = BufferView::new(cfg_bytes.to_vec().into(), vec![1], 4);
            let binds = [a_view, b_view, out_view.clone(), cfg_view];
            let wg = ((len as u32 + 255) / 256, 1, 1);
            let kernel = match call.op {
                Op::Add => &Kernel::Add,
                Op::Mul => &Kernel::Mul,
                Op::Sub => &Kernel::Sub,
                Op::Div => &Kernel::Div,
                Op::Where => &Kernel::Where,
                Op::Exp => &Kernel::Exp,
                Op::Log => &Kernel::Log,
                Op::Tanh => &Kernel::Tanh,
            };
            let bytes = backend.dispatch(kernel, &binds, [wg.0, wg.1, wg.2])?;
            if let Some(out_bytes) = bytes.get(0) {
                let slice: &[f32] = bytemuck::cast_slice(out_bytes);
                call.out.data.clone_from_slice(slice);
            }
        }
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

    pub fn forward(&self, x: &Tensor, g: &mut Graph) -> Tensor {
        let mut y = vec![0f32; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = self.b.data[o];
            for i in 0..self.in_dim {
                sum += self.w.data[o * self.in_dim + i] * x.data[i];
            }
            y[o] = sum;
        }
        Tensor::from_vec(vec![self.out_dim], y)
    }
}
