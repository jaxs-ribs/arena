use ml::graph::Graph;
use ml::Tensor;
use std::collections::HashMap;

#[test]
fn graph_run_matches_cpu() {
    let mut g = Graph::new();
    let mut tensors = HashMap::new();

    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]);
    let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]);
    tensors.insert(a.id, a.clone());
    tensors.insert(b.id, b.clone());

    let c = a.add(&b, &mut g, &mut tensors);
    let d = c.mul(&b, &mut g, &mut tensors);
    let e = d.reduce_sum(&mut g, &mut tensors);

    let expected_c = c.data.clone();
    let expected_d = d.data.clone();
    let expected_e = e.data.clone();

    tensors.get_mut(&c.id).unwrap().data.fill(0.0);
    tensors.get_mut(&d.id).unwrap().data.fill(0.0);
    tensors.get_mut(&e.id).unwrap().data.fill(0.0);

    g.run(&mut tensors).unwrap();

    assert_eq!(tensors.get(&c.id).unwrap().data, expected_c);
    assert_eq!(tensors.get(&d.id).unwrap().data, expected_d);
    assert_eq!(tensors.get(&e.id).unwrap().data, expected_e);
}

#[test]
fn graph_run_relu() {
    let mut g = Graph::new();
    let mut tensors = HashMap::new();

    let a = Tensor::from_vec(vec![3], vec![-1.0, 0.5, 2.0]);
    tensors.insert(a.id, a.clone());

    let b = a.relu(&mut g, &mut tensors);

    let expected = b.data.clone();

    tensors.get_mut(&b.id).unwrap().data.fill(0.0);

    g.run(&mut tensors).unwrap();

    assert_eq!(tensors.get(&b.id).unwrap().data, expected);
}
