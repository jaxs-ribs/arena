use compute::{BufferView, ComputeBackend, ComputeError, Kernel};
use std::sync::Arc;
use bytemuck;
use rand::{distributions::Uniform, Rng};

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

    pub fn run<B: ComputeBackend + ?Sized>(&mut self, backend: &B) -> Result<(), ComputeError> {
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

    pub fn xavier(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0f32 / (in_dim as f32 + out_dim as f32)).sqrt();
        let dist = Uniform::new(-limit, limit);
        let weights: Vec<f32> = (0..in_dim * out_dim).map(|_| rng.sample(dist)).collect();
        let bias = vec![0.0; out_dim];
        Self::new(weights, bias, in_dim, out_dim)
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

// Trait representing a differentiable layer
pub trait Layer {
    fn forward(&self, x: &Tensor, g: &mut Graph) -> Tensor;
    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>);
    fn params(&self) -> Vec<&Tensor> { Vec::new() }
    fn params_mut(&mut self) -> Vec<&mut Tensor> { Vec::new() }
}

impl Layer for Dense {
    fn forward(&self, x: &Tensor, g: &mut Graph) -> Tensor { self.forward(x, g) }
    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let (dx, dw, db) = self.backward(x, grad);
        (dx, vec![dw, db])
    }
    fn params(&self) -> Vec<&Tensor> { vec![&self.w, &self.b] }
    fn params_mut(&mut self) -> Vec<&mut Tensor> { vec![&mut self.w, &mut self.b] }
}

impl Dense {
    pub fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Tensor, Tensor) {
        let mut grad_input = vec![0.0; self.in_dim];
        let mut grad_w = vec![0.0; self.in_dim * self.out_dim];
        let mut grad_b = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let go = grad.data[o];
            for i in 0..self.in_dim {
                grad_w[o * self.in_dim + i] += go * x.data[i];
                grad_input[i] += self.w.data[o * self.in_dim + i] * go;
            }
            grad_b[o] += go;
        }
        (
            Tensor::from_vec(vec![self.in_dim], grad_input),
            Tensor::from_vec(vec![self.out_dim, self.in_dim], grad_w),
            Tensor::from_vec(vec![self.out_dim], grad_b),
        )
    }
}

#[derive(Default)]
pub struct Relu;

impl Layer for Relu {
    fn forward(&self, x: &Tensor, _g: &mut Graph) -> Tensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.max(0.0)).collect();
        Tensor::from_vec(x.shape.clone(), data)
    }

    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let data: Vec<f32> = x
            .data
            .iter()
            .zip(&grad.data)
            .map(|(&v, &g)| if v > 0.0 { g } else { 0.0 })
            .collect();
        (Tensor::from_vec(x.shape.clone(), data), Vec::new())
    }
}

#[derive(Default)]
pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, x: &Tensor, _g: &mut Graph) -> Tensor {
        let data: Vec<f32> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        Tensor::from_vec(x.shape.clone(), data)
    }

    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let forward = self.forward(x, &mut Graph::default());
        let data: Vec<f32> = forward
            .data
            .iter()
            .zip(&grad.data)
            .map(|(&s, &g)| g * s * (1.0 - s))
            .collect();
        (Tensor::from_vec(x.shape.clone(), data), Vec::new())
    }
}

#[derive(Default)]
pub struct TanhAct;

impl Layer for TanhAct {
    fn forward(&self, x: &Tensor, _g: &mut Graph) -> Tensor {
        let data: Vec<f32> = x.data.iter().map(|&v| v.tanh()).collect();
        Tensor::from_vec(x.shape.clone(), data)
    }

    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let forward = self.forward(x, &mut Graph::default());
        let data: Vec<f32> = forward
            .data
            .iter()
            .zip(&grad.data)
            .map(|(&t, &g)| g * (1.0 - t * t))
            .collect();
        (Tensor::from_vec(x.shape.clone(), data), Vec::new())
    }
}

pub struct Softmax;

impl Layer for Softmax {
    fn forward(&self, x: &Tensor, _g: &mut Graph) -> Tensor {
        let m = x.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = x.data.iter().map(|&v| (v - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let data: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
        Tensor::from_vec(x.shape.clone(), data)
    }

    fn backward(&self, x: &Tensor, grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let sm = self.forward(x, &mut Graph::default());
        let n = sm.data.len();
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                let delta = if i == j { 1.0 } else { 0.0 };
                result[j] += grad.data[i] * sm.data[i] * (delta - sm.data[j]);
            }
        }
        (Tensor::from_vec(x.shape.clone(), result), Vec::new())
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    #[must_use]
    pub fn new() -> Self { Self { layers: Vec::new() } }

    pub fn push<L: Layer + 'static>(&mut self, layer: L) { self.layers.push(Box::new(layer)); }

    pub fn forward(&self, x: &Tensor, g: &mut Graph) -> (Tensor, Vec<Tensor>) {
        let mut out = x.clone();
        let mut activations = vec![out.clone()];
        for layer in &self.layers {
            out = layer.forward(&out, g);
            activations.push(out.clone());
        }
        (out, activations)
    }

    pub fn backward(&self, activations: &[Tensor], grad: &Tensor) -> (Tensor, Vec<Tensor>) {
        let mut grad_out = grad.clone();
        let mut param_grads = Vec::new();
        for (layer, activation) in self
            .layers
            .iter()
            .rev()
            .zip(activations.iter().rev().skip(1))
        {
            let (g_in, mut p) = layer.backward(activation, &grad_out);
            grad_out = g_in;
            param_grads.extend(p);
        }
        (grad_out, param_grads)
    }

    pub fn params_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out = Vec::new();
        for layer in &mut self.layers {
            out.extend(layer.params_mut());
        }
        out
    }
}

pub struct Sgd { pub lr: f32 }

impl Sgd {
    #[must_use]
    pub fn new(lr: f32) -> Self { Self { lr } }

    pub fn step(&self, params: &mut [(&mut Tensor, &Tensor)]) {
        for (p, g) in params {
            for (pv, gv) in p.data.iter_mut().zip(&g.data) {
                *pv -= self.lr * gv;
            }
        }
    }
}

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: Vec::new(), v: Vec::new() }
    }

    pub fn step(&mut self, params: &mut [(&mut Tensor, &Tensor)]) {
        if self.m.is_empty() {
            self.m = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
            self.v = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
        }
        self.t += 1;
        for ((p, g), (m_vec, v_vec)) in params.iter_mut().zip(self.m.iter_mut().zip(self.v.iter_mut())) {
            for i in 0..p.len() {
                m_vec[i] = self.beta1 * m_vec[i] + (1.0 - self.beta1) * g.data[i];
                v_vec[i] = self.beta2 * v_vec[i] + (1.0 - self.beta2) * g.data[i] * g.data[i];
                let m_hat = m_vec[i] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v_vec[i] / (1.0 - self.beta2.powi(self.t as i32));
                p.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

pub struct PolicyNetwork {
    pub net: Sequential,
}

impl PolicyNetwork {
    #[must_use]
    pub fn new(net: Sequential) -> Self { Self { net } }

    pub fn act(&self, obs: &Tensor) -> Tensor {
        let mut g = Graph::default();
        self.net.forward(obs, &mut g).0
    }
}

pub struct ValueNetwork {
    pub net: Sequential,
}

impl ValueNetwork {
    #[must_use]
    pub fn new(net: Sequential) -> Self { Self { net } }

    pub fn value(&self, obs: &Tensor) -> Tensor {
        let mut g = Graph::default();
        self.net.forward(obs, &mut g).0
    }
}

pub fn policy_gradient_loss(log_probs: &Tensor, advantages: &Tensor) -> f32 {
    let n = log_probs.len();
    let sum: f32 = log_probs
        .data
        .iter()
        .zip(&advantages.data)
        .map(|(&l, &a)| -l * a)
        .sum();
    sum / n as f32
}

pub fn value_loss(pred: &Tensor, targets: &Tensor) -> f32 {
    let n = pred.len();
    let sum: f32 = pred
        .data
        .iter()
        .zip(&targets.data)
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    sum / n as f32
}

