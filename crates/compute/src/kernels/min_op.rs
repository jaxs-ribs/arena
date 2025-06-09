use crate::{BufferView, ComputeError};

/// Computes the pairwise minimum of two buffers.
///
/// Bindings `[a, b, output_placeholder, config]` must all be `f32` arrays of the
/// same shape. The minimum of each pair is returned in a single output buffer.
pub fn handle_min(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        // IN1, IN2, OUT, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Min kernel expects 4 buffers (input_a, input_b, output_placeholder, config)",
        ));
    }
    let input_a_view = &binds[0];
    let input_b_view = &binds[1];

    if input_a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || input_b_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Min kernel currently only supports f32 data for both inputs",
        ));
    }

    if input_a_view.data.len() != input_b_view.data.len() {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Min must have the same byte length",
        ));
    }
    if input_a_view.shape != input_b_view.shape {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Min must have the same shape",
        ));
    }

    let input_a_values: &[f32] = bytemuck::cast_slice(&input_a_view.data);
    let input_b_values: &[f32] = bytemuck::cast_slice(&input_b_view.data);

    let output_values: Vec<f32> = input_a_values
        .iter()
        .zip(input_b_values.iter())
        .map(|(&a, &b)| a.min(b))
        .collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_min() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, -1.0, 0.0, 5.0, -5.0];
        let a_bytes = Arc::from(bytemuck::cast_slice(&a_data));
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let b_data = vec![2.0f32, -2.0, 0.0, 4.0, -4.0];
        let b_bytes = Arc::from(bytemuck::cast_slice(&b_data));
        let b = BufferView::new(b_bytes, vec![b_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 5];
        let out_bytes = Arc::from(bytemuck::cast_slice(&out_data));
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes = Arc::from(bytemuck::cast_slice(&config_data));
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, b, out, config];
        let result_buffers = cpu.dispatch(&Kernel::Min, &dispatch_binds, [1, 1, 1]).unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[1.0, -2.0, 0.0, 4.0, -5.0]);
    }
}
