#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    fn test_unary_op(kernel: Kernel, input: Vec<f32>, expected: Vec<f32>) {
        let cpu = CpuBackend::new();

        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let input_buffer = BufferView::new(input_bytes, vec![input.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; expected.len()];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let output_placeholder = BufferView::new(out_bytes, vec![expected.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![input_buffer, output_placeholder, config];
        let result_buffers = cpu.dispatch(&kernel, &dispatch_binds, [1, 1, 1]).unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);

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
