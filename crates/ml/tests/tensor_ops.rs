use ml::{Graph, Tensor, Op, OpCall};
use compute::default_backend;

fn run_single(call: OpCall) -> Tensor {
    let mut g = Graph::default();
    g.add_call(call);
    let backend = default_backend();
    g.run(backend.as_ref()).unwrap();
    g.calls.pop().unwrap().out
}

#[test]
fn test_add_mul_where() {
    let a = Tensor::from_vec(vec![128], vec![1.0;128]);
    let b = Tensor::from_vec(vec![128], vec![1.0;128]);
    let out = Tensor::from_vec(vec![128], vec![0.0;128]);
    let res = run_single(OpCall{ op: Op::Add, a: a.clone(), b: b.clone(), out });
    let host_ref: Vec<f32> = a.data.iter().zip(&b.data).map(|(x,y)| x+y).collect();
    let max_diff = res
        .data
        .iter()
        .zip(&host_ref)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(max_diff < 1e-6);

    let out2 = Tensor::from_vec(vec![128], vec![0.0;128]);
    let res2 = run_single(OpCall{ op: Op::Mul, a: a.clone(), b: b.clone(), out: out2 });
    let host_ref2: Vec<f32> = a.data.iter().zip(&b.data).map(|(x,y)| x*y).collect();
    let max_diff2 = res2
        .data
        .iter()
        .zip(&host_ref2)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(max_diff2 < 1e-6);

    let mask_vals: Vec<f32> = (0..128).map(|i| if i%2==0 {1.0} else {0.0}).collect();
    let mask = Tensor::from_vec(vec![128], mask_vals.clone());
    let out3 = Tensor::from_vec(vec![128], vec![0.0;128]);
    let res3 = run_single(OpCall{ op: Op::Where, a: a, b: mask, out: out3 });
    let host_ref3: Vec<f32> = mask_vals.iter().map(|&m| if m==0.0 {1.0} else {m}).collect();
    let max_diff3 = res3
        .data
        .iter()
        .zip(&host_ref3)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(max_diff3 < 1e-6);
}

#[test]
fn test_sub_div_exp() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec(vec![4], vec![4.0, 3.0, 2.0, 1.0]);
    let out = Tensor::from_vec(vec![4], vec![0.0;4]);
    let res = run_single(OpCall{ op: Op::Sub, a: a.clone(), b: b.clone(), out });
    let expected: Vec<f32> = a.data.iter().zip(&b.data).map(|(x,y)| x - y).collect();
    for (o,e) in res.data.iter().zip(expected) { assert!((o-e).abs() < 1e-6); }

    let out2 = Tensor::from_vec(vec![4], vec![0.0;4]);
    let res2 = run_single(OpCall{ op: Op::Div, a: a.clone(), b: b.clone(), out: out2 });
    let expected2: Vec<f32> = a.data.iter().zip(&b.data).map(|(x,y)| x / y).collect();
    for (o,e) in res2.data.iter().zip(expected2) { assert!((o-e).abs() < 1e-6); }

    let out3 = Tensor::from_vec(vec![4], vec![0.0;4]);
    let res3 = run_single(OpCall{ op: Op::Exp, a: a.clone(), b: Tensor::from_vec(vec![4], vec![0.0;4]), out: out3 });
    let expected3: Vec<f32> = a.data.iter().map(|x| x.exp()).collect();
    for (o,e) in res3.data.iter().zip(expected3) { assert!((o-e).abs() < 1e-6); }
}
