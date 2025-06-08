use ml::graph::Graph;
use ml::tape::Tape;
use ml::tensor::Tensor;
use std::collections::HashMap;

fn simulate_fd(y0: f32, v0: f32, dt: f32) -> f32 {
    let mut tensors = HashMap::new();
    let y = Tensor::from_vec(vec![1], vec![y0]);
    let v = Tensor::from_vec(vec![1], vec![v0]);
    tensors.insert(y.id, y.clone());
    tensors.insert(v.id, v.clone());
    let mut g = Graph::new();

    let vdt = v.mul_scalar(dt, &mut g, &mut tensors);
    let pos = y.add(&vdt, &mut g, &mut tensors);
    let gravity = Tensor::from_vec(vec![1], vec![-9.81 * 0.5 * dt * dt]);
    tensors.insert(gravity.id, gravity.clone());
    let final_pos = pos.add(&gravity, &mut g, &mut tensors);

    final_pos.data[0]
}

#[test]
fn sphere_under_gravity_gradients() {
    let dt = 0.1_f32;
    let mut tensors = HashMap::new();

    let mut y0 = Tensor::from_vec(vec![1], vec![10.0]);
    let mut v0 = Tensor::from_vec(vec![1], vec![1.0]);
    y0.set_requires_grad();
    v0.set_requires_grad();
    tensors.insert(y0.id, y0.clone());
    tensors.insert(v0.id, v0.clone());

    let mut tape = Tape::new();
    let vdt = v0.mul_scalar(dt, &mut tape, &mut tensors);
    let pos = y0.add(&vdt, &mut tape, &mut tensors);
    let gravity = Tensor::from_vec(vec![1], vec![-9.81 * 0.5 * dt * dt]);
    tensors.insert(gravity.id, gravity.clone());
    let final_pos = pos.add(&gravity, &mut tape, &mut tensors);
    let loss = final_pos.reduce_sum(&mut tape, &mut tensors);

    tape.backward(&loss, &mut tensors).unwrap();

    let grad_y = tensors.get(&y0.id).unwrap().grad.as_ref().unwrap()[0];
    let grad_v = tensors.get(&v0.id).unwrap().grad.as_ref().unwrap()[0];

    let eps = 1e-3;
    let num_grad_y = (simulate_fd(y0.data[0] + eps, v0.data[0], dt)
        - simulate_fd(y0.data[0] - eps, v0.data[0], dt))
        / (2.0 * eps);
    let num_grad_v = (simulate_fd(y0.data[0], v0.data[0] + eps, dt)
        - simulate_fd(y0.data[0], v0.data[0] - eps, dt))
        / (2.0 * eps);

    assert!((grad_y - num_grad_y).abs() < 1e-3, "grad_y {grad_y} numeric {num_grad_y}");
    assert!((grad_v - num_grad_v).abs() < 1e-3, "grad_v {grad_v} numeric {num_grad_v}");
}

