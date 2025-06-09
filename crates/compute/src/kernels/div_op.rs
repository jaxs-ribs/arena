use crate::{BufferView, ComputeError};

/// Element-wise division of two buffers.
///
/// Bindings must contain `[a, b, output_placeholder, config]` with matching
/// shapes. Division is performed on `f32` values and the result is returned in a
/// single output buffer.
pub fn handle_div(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        // IN1, IN2, OUT, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Div kernel expects 4 buffers (input_a, input_b, output_placeholder, config)",
        ));
    }
    let input_a_view = &binds[0];
    let input_b_view = &binds[1];

    if input_a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || input_b_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Div kernel currently only supports f32 data for both inputs",
        ));
    }

    if input_a_view.data.len() != input_b_view.data.len() {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Div must have the same byte length",
        ));
    }
    if input_a_view.shape != input_b_view.shape {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Div must have the same shape",
        ));
    }

    let input_a_values: &[f32] = bytemuck::cast_slice(&input_a_view.data);
    let input_b_values: &[f32] = bytemuck::cast_slice(&input_b_view.data);

    let output_values: Vec<f32> = input_a_values
        .iter()
        .zip(input_b_values.iter())
        .map(|(&a, &b)| a / b)
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
    fn test_div() {
        let cpu = CpuBackend::new();

        let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let b_data = vec![2.0f32, 5.0, 3.0, 4.0];
        let b_bytes: Arc<[u8]> = bytemuck::cast_slice(&b_data).to_vec().into();
        let b = BufferView::new(b_bytes, vec![b_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 4];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, b, out, config];
        let result_buffers = cpu
            .dispatch(&Kernel::Div, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[5.0, 4.0, 10.0, 10.0]);
    }
}
