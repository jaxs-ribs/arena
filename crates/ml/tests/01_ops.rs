mod common;
use common::mock_backend;
use ml::graph::Graph;

#[test]
fn tensor_ops_gpu_parity() {
    use ml::*;
    let a = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec(vec![4], vec![4.0, 3.0, 2.0, 1.0]);
    let mut g = Graph::new();
    let c = a.add(&b, &mut g);          // gpu
    g.run(&mock_backend()).unwrap();
    for i in 0..4 {
        assert!((c.data()[i] - (a.data()[i] + b.data()[i])).abs() < 1e-6);
    }
} 