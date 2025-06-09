use crate::{BufferView, ComputeError};

/// Computes the natural logarithm of each input element.
///
/// Expects bindings `[input, output_placeholder, config]` using `f32` values.
/// Returns one buffer containing the logarithms of the input.
pub fn handle_log(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        // IN, OUT_placeholder, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Log kernel expects 3 buffers (input, output_placeholder, config)",
        ));
    }
    let input_view = &binds[0];

    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Log kernel currently only supports f32 data",
        ));
    }

    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| x.ln()).collect(); // Natural logarithm
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_log() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 4];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, out, config];
        let result_buffers = cpu
            .dispatch(&Kernel::Log, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        let expected = &[0.0, 0.6931472, 1.0986123, 1.3862944];
        for (i, val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }
}
