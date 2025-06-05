use ml::graph::Graph;
use ml::nn::Dense;
use ml::*;
use std::collections::HashMap;

#[test]
fn dense_forward_exact() {
    let mut tensors = HashMap::new();
    let w = vec![
        1.0, 0.5, -0.5, -1.0, // First row
        0.2, 0.3, 0.1, 0.9, // Second row
    ];
    let b = vec![0.1, -0.2];
    let layer = Dense::new(w, b.clone(), 4, 2);
    let x = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let y = layer.forward(&x, &mut Graph::new(), &mut tensors);

    let expected_y0 = 1.0 * 1.0 + 0.5 * 2.0 - 0.5 * 3.0 - 1.0 * 4.0 + b[0];
    let expected_y1 = 0.2 * 1.0 + 0.3 * 2.0 + 0.1 * 3.0 + 0.9 * 4.0 + b[1];

    assert!((y.data()[0] - expected_y0).abs() < 1e-6);
    assert!((y.data()[1] - expected_y1).abs() < 1e-6);
}
