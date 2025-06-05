use crate::{BufferView, ComputeError};

pub fn handle_rng_normal(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 2 {
        return Err(ComputeError::ShapeMismatch("RngNormal kernel expects 2 buffers (output_placeholder, config)"));
    }
    let output_view = &binds[0];
    if output_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch("RngNormal kernel currently only supports f32 output"));
    }
    let num_values_to_generate = output_view.shape.iter().product::<usize>();
    let output_values: Vec<f32> = (0..num_values_to_generate).map(|i| i as f32 * 0.1).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockCpu, ComputeBackend, Kernel};
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_rng_normal_produces_deterministic_sequence() {
        let cpu = MockCpu::default();
        let num_values_to_generate = 5usize;
        let expected_output_data: Vec<f32> = (0..num_values_to_generate).map(|i| i as f32 * 0.1).collect();

        let output_placeholder: StdArc<[u8]> = vec![0u8; num_values_to_generate * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(output_placeholder, vec![num_values_to_generate], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: StdArc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = [output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::RngNormal, &dispatch_binds, [1,1,1]).expect("Dispatch for RngNormal failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());
        for (i, (got, expected)) in output_values.iter().zip(expected_output_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, "Mismatch for RngNormal at index {}. Got: {}, Expected: {}", i, got, expected);
        }
    }
}
