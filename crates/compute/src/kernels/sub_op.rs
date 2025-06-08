use crate::{BufferView, ComputeError};

pub fn handle_sub(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        // IN1, IN2, OUT, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Sub kernel expects 4 buffers (input_a, input_b, output_placeholder, config)",
        ));
    }
    let input_a_view = &binds[0];
    let input_b_view = &binds[1];

    if input_a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || input_b_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Sub kernel currently only supports f32 data for both inputs",
        ));
    }

    if input_a_view.data.len() != input_b_view.data.len() {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Sub must have the same byte length",
        ));
    }
    if input_a_view.shape != input_b_view.shape {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Sub must have the same shape",
        ));
    }

    let input_a_values: &[f32] = bytemuck::cast_slice(&input_a_view.data);
    let input_b_values: &[f32] = bytemuck::cast_slice(&input_b_view.data);

    let output_values: Vec<f32> = input_a_values
        .iter()
        .zip(input_b_values.iter())
        .map(|(&a, &b)| a - b)
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
    fn test_sub() {
        let cpu = CpuBackend::new();

        let a = BufferView::from(Arc::new(vec![1.0, 2.0, 3.0, 4.0]));
        let b = BufferView::from(Arc::new(vec![5.0, 6.0, 7.0, 8.0]));
        let out = BufferView::new(Arc::new(vec![0.0; 4]), ());

        let dispatch_binds = &[&a, &b, &out];
        let result_buffers = cpu
            .dispatch(&Kernel::Sub, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
        assert_eq!(result, &[-4.0, -4.0, -4.0, -4.0]);
    }
}
