use ml::*;
use ml::tape::Tape;
use ml::nn::Dense;
use std::collections::HashMap;

fn finite_diff_check<F>(
    w_with_grad: &Tensor,
    x: &Tensor,
    f: &F,
    epsilon: f32,
) where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    let mut w_plus = w_with_grad.clone();
    let mut w_minus = w_with_grad.clone();

    for i in 0..w_plus.data.len() {
        w_plus.data[i] += epsilon;
        w_minus.data[i] -= epsilon;

        let loss_plus = f(&w_plus, x);
        let loss_minus = f(&w_minus, x);

        let numerical_grad = (loss_plus.data[0] - loss_minus.data[0]) / (2.0 * epsilon);
        let analytical_grad = w_with_grad.grad.as_ref().unwrap()[i];

        let diff = (numerical_grad - analytical_grad).abs();
        assert!(diff < 1e-3, "Grad check failed for weight {i}. Numerical: {numerical_grad}, Analytical: {analytical_grad}");

        w_plus.data[i] -= epsilon;
        w_minus.data[i] += epsilon;
    }
}

#[test]
fn dense_backward_fd() {
    let mut tensors = HashMap::new();

    let mut layer = Dense::random(3, 2, 0);
    layer.w = layer.w.clone().with_grad();
    layer.b = layer.b.clone().with_grad();
    tensors.insert(layer.w.id, layer.w.clone());
    tensors.insert(layer.b.id, layer.b.clone());

    let x = Tensor::from_vec(vec![3], vec![0.9, -0.1, 0.3]);
    tensors.insert(x.id, x.clone());

    let mut tape = Tape::new();
    let (y, wx) = layer.forward(&x, &mut tape);
    tensors.insert(y.id, y.clone());
    tensors.insert(wx.id, wx.clone());

    let loss = y.reduce_sum(&mut tape);
    tensors.insert(loss.id, loss.clone());

    tape.backward(&loss, &mut tensors).unwrap();

    let updated_w = tensors.get(&layer.w.id).unwrap();

    finite_diff_check(updated_w, &x, &|w,x| layer.fd_loss(w,x), 1e-3);
} 