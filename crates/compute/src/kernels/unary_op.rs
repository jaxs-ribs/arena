#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    fn test_unary_op(kernel: Kernel, input: Vec<f32>, expected: Vec<f32>) {
        let cpu = CpuBackend::new();

        let input_buffer = BufferView::from(Arc::new(input));
        let output_placeholder = BufferView::new(Arc::new(vec![0.0f32; expected.len()]), ());

        let dispatch_binds = &[&input_buffer, &output_placeholder];
        let result_buffers = cpu.dispatch(&kernel, &dispatch_binds, [1, 1, 1]).unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neg() {
        test_unary_op(
            Kernel::Neg,
            vec![1.0, -2.0, 3.0],
            vec![-1.0, 2.0, -3.0],
        );
    }

    #[test]
    fn test_exp() {
        test_unary_op(
            Kernel::Exp,
            vec![1.0, 0.0, -1.0],
            vec![1.0f32.exp(), 0.0f32.exp(), (-1.0f32).exp()],
        );
    }

    #[test]
    fn test_log() {
        test_unary_op(
            Kernel::Log,
            vec![1.0, 2.0, 3.0],
            vec![1.0f32.ln(), 2.0f32.ln(), 3.0f32.ln()],
        );
    }

    #[test]
    fn test_sqrt() {
        test_unary_op(
            Kernel::Sqrt,
            vec![1.0, 4.0, 9.0],
            vec![1.0, 2.0, 3.0],
        );
    }

    #[test]
    fn test_rsqrt() {
        test_unary_op(
            Kernel::Rsqrt,
            vec![1.0, 4.0, 9.0],
            vec![1.0, 0.5, 1.0 / 3.0],
        );
    }
} 