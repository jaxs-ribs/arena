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

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_clamp() {
        let cpu = CpuBackend::new();

        let a = BufferView::from(Arc::new(vec![-10.0, 10.0, 0.0, 0.5, 0.7]));
        let min = BufferView::from(Arc::new(vec![0.0]));
        let max = BufferView::from(Arc::new(vec![1.0]));
        let out = BufferView::new(Arc::new(vec![0.0; 5]), ());

        let dispatch_binds = &[&a, &min, &max, &out];
        let result_buffers = cpu
            .dispatch(&Kernel::Clamp, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
        assert_eq!(result, &[0.0, 1.0, 0.0, 0.5, 0.7]);
    }
}
