use crate::{BufferView, ComputeError};

pub fn handle_clamp(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 5 {
        // value, min_val, max_val, out_placeholder, config per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Clamp kernel expects 5 buffers",
        ));
    }
    let value_view = &binds[0];
    let min_view = &binds[1];
    let max_view = &binds[2];
    // binds[3] is output_placeholder, binds[4] is config

    if value_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || min_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || max_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Clamp kernel currently only supports f32 data for all inputs",
        ));
    }

    // Ensure all inputs have the same number of elements and shape
    if !(value_view.data.len() == min_view.data.len() && min_view.data.len() == max_view.data.len())
    {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Clamp must have the same byte length",
        ));
    }
    if !(value_view.shape == min_view.shape && min_view.shape == max_view.shape) {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Clamp must have the same shape",
        ));
    }

    let value_values: &[f32] = bytemuck::cast_slice(&value_view.data);
    let min_values: &[f32] = bytemuck::cast_slice(&min_view.data);
    let max_values: &[f32] = bytemuck::cast_slice(&max_view.data);

    let output_values: Vec<f32> = value_values
        .iter()
        .zip(min_values.iter())
        .zip(max_values.iter())
        .map(|((&val, &min_val), &max_val)| val.max(min_val).min(max_val)) // clamp operation
        .collect();

    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use crate::{BufferView, Kernel, CpuBackend};
    use std::sync::Arc;

    #[test]
    fn mock_clamp_clamps_values() {
        let cpu = CpuBackend::new();
        let value_data = vec![1.0f32, -2.0, 0.0, 3.5, -0.5, 7.0, 0.5];
        let min_values_data = vec![0.0f32, 0.0, 0.1, -1.0, 0.0, 6.0, -1.0];
        let max_values_data = vec![1.0f32, 1.0, 0.1, 2.0, 1.0, 6.5, 0.0];

        let expected_output_data: Vec<f32> = value_data
            .iter()
            .zip(min_values_data.iter())
            .zip(max_values_data.iter())
            .map(|((&val, &min_val), &max_val)| val.max(min_val).min(max_val))
            .collect();

        let value_bytes: Arc<[u8]> = bytemuck::cast_slice(&value_data).to_vec().into();
        let value_buffer_view = BufferView::new(
            value_bytes,
            vec![value_data.len()],
            std::mem::size_of::<f32>(),
        );

        let min_values_bytes: Arc<[u8]> = bytemuck::cast_slice(&min_values_data).to_vec().into();
        let min_values_buffer_view = BufferView::new(
            min_values_bytes,
            vec![min_values_data.len()],
            std::mem::size_of::<f32>(),
        );

        let max_values_bytes: Arc<[u8]> = bytemuck::cast_slice(&max_values_data).to_vec().into();
        let max_values_buffer_view = BufferView::new(
            max_values_bytes,
            vec![max_values_data.len()],
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
                &Kernel::Clamp,
                &[
                    value_buffer_view,
                    min_values_buffer_view,
                    max_values_buffer_view,
                    output_buffer_view,
                    config_buffer_view,
                ],
                workgroups,
            )
            .expect("Dispatch for Clamp failed");

        assert_eq!(
            result_buffers.len(),
            1,
            "Clamp should return one output buffer"
        );
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_output_data.len() * std::mem::size_of::<f32>()
        );

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (i, (got, expected)) in output_values
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Clamp at index {}. Got: {}, Expected: {}. Input val: {}, min: {}, max: {}", 
                    i, got, expected, value_data[i], min_values_data[i], max_values_data[i]);
        }
    }
}
