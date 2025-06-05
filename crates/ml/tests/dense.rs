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

#[test]
fn dense_xavier_init_stats() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(42);
    let dense = Dense::xavier(4, 3, &mut rng);
    assert_eq!(dense.w.shape, vec![3, 4]);
    assert_eq!(dense.b.shape, vec![3]);
    assert_eq!(dense.w.data.len(), 12);
    assert_eq!(dense.b.data.len(), 3);

    let mean: f32 = dense.w.data.iter().copied().sum::<f32>() / dense.w.data.len() as f32;
    assert!(mean.abs() < 0.1);
    let var: f32 = dense.w.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / dense.w.data.len() as f32;
    let limit = (6.0f32 / (4.0 + 3.0)).sqrt();
    let expected_var = limit * limit / 3.0;
    let rel_diff = (var - expected_var).abs() / expected_var;
    assert!(rel_diff < 0.5);
}


