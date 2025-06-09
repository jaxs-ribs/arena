use crate::{BufferView, ComputeError};

/// Applies the exponential function to each element of the input buffer.
///
/// Bindings must be `[input, output_placeholder, config]` containing `f32`
/// values. The computed exponentials are returned in a single output buffer.
pub fn handle_exp(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        // IN, OUT_placeholder, CONFIG per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Exp kernel expects 3 buffers (input, output_placeholder, config)",
        ));
    }
    let input_view = &binds[0];
    // binds[1] is output_placeholder, binds[2] is config

    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Exp kernel currently only supports f32 data",
        ));
    }

    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| x.exp()).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_exp() {
        let cpu = CpuBackend::new();

        let a_data = vec![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 5];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, out, config];
        let result_buffers = cpu
            .dispatch(&Kernel::Exp, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        let expected = &[
            1.0,
            2.7182817,
            0.36787945,
            7.389056,
            0.13533528,
        ];
        for (i, val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }
}
