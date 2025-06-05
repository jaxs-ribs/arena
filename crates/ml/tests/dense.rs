use ml::{Dense, Tensor, Graph};

#[test]
fn dense_forward_known_case() {
    let w = vec![1.0, 0.5,
                 -1.0, 2.0,
                 0.2, 0.2];
    let b = vec![0.1, -0.1];
    let x = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
    let dense = Dense::new(w, b.clone(), 3, 2);
    let mut g = Graph::default();
    let y = dense.forward(&x, &mut g);
    assert!((y.data[0] - (-0.9)).abs() < 1e-6);
    assert!((y.data[1] - 2.9).abs() < 1e-6);
}

#[test]
fn dense_bias_only() {
    let w = vec![0.0, 0.0,
                 0.0, 0.0,
                 0.0, 0.0];
    let b = vec![0.5, -0.5];
    let x = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
    let dense = Dense::new(w, b.clone(), 3, 2);
    let mut g = Graph::default();
    let y = dense.forward(&x, &mut g);
    assert!((y.data[0] - 0.5).abs() < 1e-6);
    assert!((y.data[1] + 0.5).abs() < 1e-6);
}


