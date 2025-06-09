use crate::{BufferView, ComputeError};

/// Calculates the mean of all elements in the input buffer.
///
/// Bindings `[input, output_placeholder, config]` must use `f32` values. The
/// resulting mean is written to a single buffer which is returned.
pub fn handle_reduce_mean(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "ReduceMean kernel expects 3 buffers",
        ));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "ReduceMean kernel currently only supports f32 input data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let count = input_values.len();
    let mean_value: f32 = if count == 0 {
        0.0
    } else {
        input_values.iter().sum::<f32>() / count as f32
    };
    let out_bytes = bytemuck::bytes_of(&mean_value).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn mock_reduce_mean_computes_mean() {
        let cpu = CpuBackend::new();

        let input_data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -2.0];
        let expected_mean1: f32 = input_data1.iter().sum::<f32>() / (input_data1.len() as f32);

        let input_bytes1: Arc<[u8]> = bytemuck::cast_slice(&input_data1).to_vec().into();
        let input_buffer_view1 = BufferView::new(
            input_bytes1,
            vec![input_data1.len()],
            std::mem::size_of::<f32>(),
        );

        let output_placeholder: Arc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder.clone(),
            vec![1],
            std::mem::size_of::<f32>(),
        );
        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes.clone(),
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let dispatch_binds1 = [
            input_buffer_view1,
            output_buffer_view.clone(),
            config_buffer_view.clone(),
        ];
        let result_buffers1 = cpu
            .dispatch(&Kernel::ReduceMean, &dispatch_binds1, [1, 1, 1])
            .expect("Dispatch for ReduceMean (case 1) failed");

        assert_eq!(result_buffers1.len(), 1);
        let output_bytes1 = &result_buffers1[0];
        assert_eq!(output_bytes1.len(), std::mem::size_of::<f32>());
        let output_value1: f32 = *bytemuck::from_bytes(output_bytes1);
        assert!((output_value1 - expected_mean1).abs() < 1e-6);

        let input_data2: Vec<f32> = Vec::new();
        let expected_mean2: f32 = 0.0;

        let input_bytes2: Arc<[u8]> = bytemuck::cast_slice(&input_data2).to_vec().into();
        let input_buffer_view2 = BufferView::new(
            input_bytes2,
            vec![input_data2.len()],
            std::mem::size_of::<f32>(),
        );
        let dispatch_binds2 = [input_buffer_view2, output_buffer_view, config_buffer_view];
        let result_buffers2 = cpu
            .dispatch(&Kernel::ReduceMean, &dispatch_binds2, [1, 1, 1])
            .expect("Dispatch for ReduceMean (case 2) failed");

        assert_eq!(result_buffers2.len(), 1);
        let output_bytes2 = &result_buffers2[0];
        assert_eq!(output_bytes2.len(), std::mem::size_of::<f32>());
        let output_value2: f32 = *bytemuck::from_bytes(output_bytes2);
        assert!((output_value2 - expected_mean2).abs() < 1e-6);
    }
}
