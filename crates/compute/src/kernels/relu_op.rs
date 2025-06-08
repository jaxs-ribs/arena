use crate::{BufferView, ComputeError};

pub fn handle_relu(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch("Relu kernel expects 3 buffers"));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Relu kernel currently only supports f32 data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| x.max(0.0)).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_relu_computes_rectified_linear_unit() {
        let cpu = CpuBackend::new();
        let input_data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 20.0, -20.0];
        let expected_output_data: Vec<f32> = input_data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

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
            .dispatch(&Kernel::Relu, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for Relu failed");

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
                "Mismatch for Relu. Got: {}, Expected: {}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_relu() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, -1.0, 0.0, 5.0, -5.0];
        let a_bytes = Arc::from(bytemuck::cast_slice(&a_data));
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 5];
        let out_bytes = Arc::from(bytemuck::cast_slice(&out_data));
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let dispatch_binds = vec![a, out];
        let result_buffers = cpu.dispatch(&Kernel::Relu, &dispatch_binds, [1, 1, 1]).unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[1.0, 0.0, 0.0, 5.0, 0.0]);
    }
}
