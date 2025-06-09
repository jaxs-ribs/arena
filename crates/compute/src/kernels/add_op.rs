use crate::{BufferView, ComputeError};

/// Element-wise addition of two buffers.
///
/// The function expects three bindings: the first two are input buffers `a` and
/// `b` containing `f32` values, while the third is a placeholder for the output
/// buffer. All inputs must have the same shape and element size. The returned
/// vector contains a single buffer with the computed sums.
pub fn handle_add(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch("Add kernel expects 3 buffers"));
    }
    let input_a_view = &binds[0];
    let input_b_view = &binds[1];
    // binds[2] is output_placeholder

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

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_add() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let b_bytes: Arc<[u8]> = bytemuck::cast_slice(&b_data).to_vec().into();
        let b = BufferView::new(b_bytes, vec![b_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 4];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let dispatch_binds = vec![a, b, out];
        let result_buffers = cpu
            .dispatch(&Kernel::Add, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }
}
