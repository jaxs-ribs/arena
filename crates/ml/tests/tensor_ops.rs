use ml::{Graph, Tensor, Op, OpCall};
use compute::MockCpu;

fn run_single(call: OpCall) -> Tensor {
    let mut g = Graph::default();
    g.add_call(call);
    g.run(&MockCpu).unwrap();
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
