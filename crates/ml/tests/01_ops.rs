use ml::graph::Graph;
use ml::recorder::Recorder;
use ml::*;
use std::collections::HashMap;

#[test]
fn tensor_ops_gpu_parity() {
    let mut g = Graph::new();
    let mut tensors = HashMap::new();

    let a = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    tensors.insert(a.id, a.clone());
    tensors.insert(b.id, b.clone());

    let c = a.add(&b, &mut g, &mut tensors);
    let d = a.mul(&b, &mut g, &mut tensors);
    let e = c.reduce_sum(&mut g, &mut tensors);
    let f = d.reduce_sum(&mut g, &mut tensors);

    assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(d.data(), &[5.0, 12.0, 21.0, 32.0]);
    assert_eq!(e.data(), &[36.0]);
    assert_eq!(f.data(), &[70.0]);

    assert_eq!(g.nodes().len(), 4);
}
