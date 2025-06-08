use ml::graph::Graph;
use ml::tape::Tape;
use ml::tensor::Tensor;
use std::collections::HashMap;

#[test]
fn relu_forward() {
    let mut tensors = HashMap::new();
    let mut g = Graph::new();
    let x = Tensor::from_vec(vec![3], vec![-1.0, 0.5, 0.0]);
    tensors.insert(x.id, x.clone());
    let y = x.relu(&mut g, &mut tensors);
    assert_eq!(y.data(), &[0.0, 0.5, 0.0]);
}

#[test]
fn relu_backward() {
    let mut tensors = HashMap::new();
    let mut x = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 2.0]);
    x.set_requires_grad();
    tensors.insert(x.id, x.clone());
    let mut tape = Tape::new();
    let y = x.relu(&mut tape, &mut tensors);
    let loss = y.reduce_sum(&mut tape, &mut tensors);
    tape.backward(&loss, &mut tensors).unwrap();
    let grad = tensors.get(&x.id).unwrap().grad.as_ref().unwrap().clone();
    assert_eq!(grad, vec![0.0, 0.0, 1.0]);
}
