use crate::{BufferView, ComputeError};

pub fn handle_reduce_sum(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "ReduceSum kernel expects 3 buffers",
        ));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "ReduceSum kernel currently only supports f32 input data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let sum_value: f32 = input_values.iter().sum();
    let out_bytes = bytemuck::bytes_of(&sum_value).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn mock_reduce_sum_computes_sum() {
        let cpu = CpuBackend::new();
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -2.0];
        let expected_sum: f32 = input_data.iter().sum();

        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes,
            vec![input_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_placeholder: Arc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view =
            BufferView::new(output_placeholder, vec![1], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view =
            BufferView::new(config_bytes, vec![1], std::mem::size_of::<u32>());

        let dispatch_binds = &[input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu
            .dispatch(&Kernel::ReduceSum, dispatch_binds, [1, 1, 1])
            .expect("Dispatch for ReduceSum failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), std::mem::size_of::<f32>());
        let output_value: f32 = *bytemuck::from_bytes(output_bytes);
        assert!(
            (output_value - expected_sum).abs() < 1e-6,
            "Mismatch for ReduceSum. Got: {}, Expected: {}",
            output_value,
            expected_sum
        );
    }
}
