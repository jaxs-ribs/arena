use crate::{BufferView, ComputeError};

/// Negates each element of the input buffer.
///
/// The bindings are `[input, output_placeholder, config]` where the input
/// buffer contains `f32` values. The negated results are returned as a single
/// buffer inside the returned vector.
pub fn handle_neg(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        // IN1, OUT, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Neg kernel expects 3 buffers (input, output_placeholder, config)",
        ));
    }
    let input_view = &binds[0];
    // binds[1] is output_placeholder, binds[2] is config

    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Neg kernel currently only supports f32 data",
        ));
    }

    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| -x).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_neg() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, -1.0, 0.0, 5.0, -5.0];
        let a_bytes = Arc::from(bytemuck::cast_slice(&a_data));
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 5];
        let out_bytes = Arc::from(bytemuck::cast_slice(&out_data));
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes = Arc::from(bytemuck::cast_slice(&config_data));
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, out, config];
        let result_buffers = cpu.dispatch(&Kernel::Neg, &dispatch_binds, [1, 1, 1]).unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[-1.0, 1.0, 0.0, -5.0, 5.0]);
    }
}
