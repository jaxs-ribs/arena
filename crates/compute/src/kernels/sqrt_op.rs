use crate::{BufferView, ComputeError};

pub fn handle_sqrt(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch("Sqrt kernel expects 3 buffers"));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Sqrt kernel currently only supports f32 data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| x.sqrt()).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use crate::{ComputeBackend, Kernel, backend::mock_cpu::MockCpu};
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_sqrt_computes_square_root() {
        let cpu = MockCpu::default();
        let input_data = vec![0.0f32, 1.0, 4.0, 9.0, 2.0];
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| x.sqrt()).collect();

        let input_bytes: StdArc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes,
            vec![input_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_placeholder: StdArc<[u8]> =
            vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder,
            vec![expected_output_data.len()],
            std::mem::size_of::<f32>(),
        );

        let config_data = vec![0u32];
        let config_bytes: StdArc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu
            .dispatch(&Kernel::Sqrt, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for Sqrt failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_output_data.len() * std::mem::size_of::<f32>()
        );
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());
        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "Mismatch for Sqrt. Got: {}, Expected: {}",
                got,
                expected
            );
        }
    }
}
