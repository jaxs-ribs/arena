use ml::{Tensor, Dense, Graph, Relu, TanhAct, Sigmoid, Softmax, Sequential, Layer, Sgd, Adam, PolicyNetwork, ValueNetwork, policy_gradient_loss, value_loss};

fn close(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x,y)| (*x - *y).abs() < 1e-5)
}

#[test]
fn dense_backward_gradients() {
    let w = vec![1.0, 2.0,
                 -3.0, 0.5];
    let b = vec![0.1, -0.2];
    let x = Tensor::from_vec(vec![2], vec![0.5, -1.0]);
    let dense = Dense::new(w.clone(), b.clone(), 2, 2);
    let mut g = Graph::default();
    let _y = dense.forward(&x, &mut g);
    let grad_out = Tensor::from_vec(vec![2], vec![1.0, -2.0]);
    let (dx, dw, db) = dense.backward(&x, &grad_out);
    let expected_dx = vec![1.0*1.0 + -3.0*(-2.0), 2.0*1.0 + 0.5*(-2.0)];
    let expected_dw = vec![0.5*1.0, -1.0*1.0,
                           0.5*(-2.0), -1.0*(-2.0)];
    assert!(close(&dx.data, &expected_dx));
    assert!(close(&dw.data, &expected_dw));
    assert!(close(&db.data, &grad_out.data));
}

#[test]
fn relu_forward_backward() {
    let x = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 2.0]);
    let relu = Relu::default();
    let y = relu.forward(&x, &mut Graph::default());
    assert_eq!(y.data, vec![0.0, 0.0, 2.0]);
    let grad_out = Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]);
    let (dx, _) = relu.backward(&x, &grad_out);
    assert_eq!(dx.data, vec![0.0, 0.0, 0.3]);
}

#[test]
fn sigmoid_forward_backward() {
    let x = Tensor::from_vec(vec![2], vec![0.0, 1.0]);
    let sig = Sigmoid::default();
    let y = sig.forward(&x, &mut Graph::default());
    let expected = vec![0.5, 1.0 / (1.0 + (-1.0f32).exp())];
    assert!(close(&y.data, &expected));
    let grad_out = Tensor::from_vec(vec![2], vec![0.5, -0.5]);
    let (dx, _) = sig.backward(&x, &grad_out);
    let s0 = 0.5f32;
    let s1 = expected[1];
    let exp_dx = vec![0.5 * s0 * (1.0 - s0), -0.5 * s1 * (1.0 - s1)];
    assert!(close(&dx.data, &exp_dx));
}

#[test]
fn tanh_forward_backward() {
    let x = Tensor::from_vec(vec![2], vec![0.0, 1.0]);
    let t = TanhAct::default();
    let y = t.forward(&x, &mut Graph::default());
    let expected = vec![0.0f32.tanh(), 1.0f32.tanh()];
    assert!(close(&y.data, &expected));
    let grad_out = Tensor::from_vec(vec![2], vec![0.2, -0.1]);
    let (dx, _) = t.backward(&x, &grad_out);
    let exp_dx = vec![0.2 * (1.0 - expected[0].powi(2)), -0.1 * (1.0 - expected[1].powi(2))];
    assert!(close(&dx.data, &exp_dx));
}

#[test]
fn softmax_forward_backward() {
    let x = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
    let sm = Softmax;
    let y = sm.forward(&x, &mut Graph::default());
    let exp_vals: Vec<f32> = vec![1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
    let sum: f32 = exp_vals.iter().sum();
    let expected: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();
    assert!(close(&y.data, &expected));
    let grad_out = Tensor::from_vec(vec![3], vec![0.1, -0.1, 0.0]);
    let (dx, _) = sm.backward(&x, &grad_out);
    // compute analytic gradient
    let mut exp_dx = vec![0.0f32;3];
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i==j {1.0} else {0.0};
            exp_dx[j] += grad_out.data[i] * expected[i] * (delta - expected[j]);
        }
    }
    assert!(close(&dx.data, &exp_dx));
}

#[test]
fn sequential_forward_backward() {
    let mut seq = Sequential::new();
    seq.push(Dense::new(vec![1.0], vec![0.0], 1, 1));
    seq.push(Relu::default());
    seq.push(Dense::new(vec![0.5], vec![0.0], 1, 1));
    let x = Tensor::from_vec(vec![1], vec![2.0]);
    let mut g = Graph::default();
    let (out, activations) = seq.forward(&x, &mut g);
    // manual
    let h1: f32 = 1.0*2.0 + 0.0;
    let h1a = h1.max(0.0);
    let y = 0.5*h1a + 0.0;
    assert!((out.data[0]-y).abs() < 1e-6);
    let grad_out = Tensor::from_vec(vec![1], vec![1.0]);
    let (dx, grads) = seq.backward(&activations, &grad_out);
    // grads contains dw1, db1, dw2, db2 in reverse
    assert_eq!(grads.len(), 4);
    // verify final grad
    let dw2 = vec![h1a*1.0];
    let db2 = vec![1.0];
    assert!(close(&grads[0].data, &dw2));
    assert!(close(&grads[1].data, &db2));
    // backward through relu and first dense
    let relu_grad = if h1>0.0 {1.0} else {0.0};
    let dw1 = vec![2.0 * 0.5 * relu_grad];
    let db1 = vec![0.5 * relu_grad];
    assert!(close(&grads[2].data, &dw1));
    assert!(close(&grads[3].data, &db1));
    assert!(close(&dx.data, &[0.5*relu_grad*1.0]));
}

#[test]
fn sgd_update() {
    let mut param = Tensor::from_vec(vec![2], vec![1.0, -1.0]);
    let grad = Tensor::from_vec(vec![2], vec![0.5, -0.5]);
    let mut opt = Sgd::new(0.1);
    opt.step(&mut [(&mut param, &grad)]);
    assert!(close(&param.data, &[0.95, -0.95]));
}

#[test]
fn adam_update() {
    let mut param = Tensor::from_vec(vec![1], vec![1.0]);
    let grad = Tensor::from_vec(vec![1], vec![0.1]);
    let mut opt = Adam::new(0.1);
    opt.step(&mut [(&mut param, &grad)]);
    // after first step of Adam with zero init moments
    let m_hat: f32 = 0.1; // (1-beta1) * grad / (1-beta1)
    let v_hat: f32 = 0.01; // (1-beta2) * grad^2 / (1-beta2)
    let expected = 1.0 - 0.1 * m_hat / (v_hat.sqrt() + 1e-8f32);
    assert!((param.data[0]-expected).abs() < 1e-6);
}

#[test]
fn policy_network_output() {
    let mut seq = Sequential::new();
    seq.push(Dense::new(vec![0.5], vec![0.0], 1, 1));
    seq.push(TanhAct::default());
    let policy = PolicyNetwork::new(seq);
    let x = Tensor::from_vec(vec![1], vec![2.0]);
    let out = policy.act(&x);
    assert_eq!(out.shape, vec![1]);
    assert!(out.data[0] <= 1.0 && out.data[0] >= -1.0);
}

#[test]
fn value_network_output() {
    let mut seq = Sequential::new();
    seq.push(Dense::new(vec![1.0], vec![0.0], 1, 1));
    let value_net = ValueNetwork::new(seq);
    let x = Tensor::from_vec(vec![1], vec![3.0]);
    let out = value_net.value(&x);
    assert_eq!(out.shape, vec![1]);
}

#[test]
fn policy_gradient_loss_calc() {
    let log_probs = Tensor::from_vec(vec![3], vec![-0.1, -0.2, -0.3]);
    let adv = Tensor::from_vec(vec![3], vec![1.0, 0.5, -0.5]);
    let loss = policy_gradient_loss(&log_probs, &adv);
    let expected = (-(-0.1*1.0 + -0.2*0.5 + -0.3*-0.5))/3.0;
    assert!((loss - expected).abs() < 1e-6);
}

#[test]
fn value_loss_calc() {
    let pred = Tensor::from_vec(vec![2], vec![1.0, 2.0]);
    let target = Tensor::from_vec(vec![2], vec![1.5, 1.5]);
    let loss = value_loss(&pred, &target);
    let expected: f32 = ((1.0f32-1.5).powi(2) + (2.0f32-1.5).powi(2)) / 2.0;
    assert!((loss - expected).abs() < 1e-6);
}

