use crate::{BufferView, ComputeError};
// std::sync::Arc is not directly used by handle_add, BufferView handles its own Arc.

// Add operation handler
pub fn handle_add(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        // IN1, IN2, OUT, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Add kernel expects 4 buffers (input_a, input_b, output_placeholder, config)",
        ));
    }
    let input_a_view = &binds[0];
    let input_b_view = &binds[1];
    // binds[2] is output_placeholder, binds[3] is config

    if input_a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || input_b_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Add kernel currently only supports f32 data for both inputs",
        ));
    }

    if input_a_view.data.len() != input_b_view.data.len() {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Add must have the same byte length",
        ));
    }
    if input_a_view.shape != input_b_view.shape {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Add must have the same shape",
        ));
    }

    let input_a_values: &[f32] = bytemuck::cast_slice(&input_a_view.data);
    let input_b_values: &[f32] = bytemuck::cast_slice(&input_b_view.data);

    let output_values: Vec<f32> = input_a_values
        .iter()
        .zip(input_b_values.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use crate::{BufferView, Kernel, CpuBackend};
    use std::sync::Arc;

    #[test]
    fn mock_add_adds_values() {
        let cpu = CpuBackend::new();
        let input_a_data = vec![1.0f32, -2.0, 0.0, 3.5, -0.5];
        let input_b_data = vec![0.5f32, 2.0, -1.0, -0.5, 10.0];
        let expected_output_data: Vec<f32> = input_a_data
            .iter()
            .zip(input_b_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let input_a_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_a_data).to_vec().into();
        let input_a_buffer_view = BufferView::new(
            input_a_bytes,
            vec![input_a_data.len()],
            std::mem::size_of::<f32>(),
        );

        let input_b_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_b_data).to_vec().into();
        let input_b_buffer_view = BufferView::new(
            input_b_bytes,
            vec![input_b_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> =
            vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes,
            vec![expected_output_data.len()],
            std::mem::size_of::<f32>(),
        );

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let workgroups = [1, 1, 1];
        let result_buffers = cpu
            .dispatch(
                &Kernel::Add,
                &[
                    input_a_buffer_view,
                    input_b_buffer_view,
                    output_buffer_view,
                    config_buffer_view,
                ],
                workgroups,
            )
            .expect("Dispatch for Add failed");

        assert_eq!(
            result_buffers.len(),
            1,
            "Add should return one output buffer"
        );
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
                "Mismatch. Got: {}, Expected: {}",
                got,
                expected
            );
        }
    }
}
