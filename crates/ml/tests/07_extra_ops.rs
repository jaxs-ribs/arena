use ml::graph::Graph;
use ml::recorder::Recorder;
use ml::*;
use std::collections::HashMap;

#[test]
fn extra_tensor_ops() {
    let mut g = Graph::new();
    let mut tensors = HashMap::new();

    let a = Tensor::from_vec(vec![2], vec![1.0, 4.0]);
    let b = Tensor::from_vec(vec![2], vec![2.0, 5.0]);
    tensors.insert(a.id, a.clone());
    tensors.insert(b.id, b.clone());

    let div = a.div(&b, &mut g, &mut tensors);
    let neg = a.neg(&mut g, &mut tensors);
    let log = a.log(&mut g, &mut tensors);
    let sqrt = b.sqrt(&mut g, &mut tensors);
    let relu = neg.relu(&mut g, &mut tensors);
    let sig = a.sigmoid(&mut g, &mut tensors);
    let max = a.max(&b, &mut g, &mut tensors);
    let max_r = a.reduce_max(&mut g, &mut tensors);

    assert_eq!(div.data(), &[0.5, 0.8]);
    assert_eq!(neg.data(), &[-1.0, -4.0]);
    for (o,e) in log.data().iter().zip([1.0f32.ln(), 4.0f32.ln()].iter()) {assert!((o-e).abs()<1e-6);}
    for (o,e) in sqrt.data().iter().zip([2.0f32.sqrt(), 5.0f32.sqrt()].iter()) {assert!((o-e).abs()<1e-6);}
    assert_eq!(relu.data(), &[0.0, 0.0]);
    for (o,e) in sig.data().iter().zip([1.0/(1.0+(-1.0f32).exp()), 1.0/(1.0+(-4.0f32).exp())].iter()) {assert!((o-e).abs()<1e-6);}
    assert_eq!(max.data(), &[2.0, 5.0]);
    assert_eq!(max_r.data(), &[4.0]);
    assert_eq!(g.nodes().len(), 8);
}

