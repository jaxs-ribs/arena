use crate::{BufferView, ComputeError};

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

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_exp_computes_exponential() {
        let cpu = CpuBackend::new();
        let input_data = vec![0.0f32, 1.0, -1.0, 2.0, std::f32::consts::LN_2];
        let expected_output_data: Vec<f32> = input_data.iter().map(|x| x.exp()).collect();

        let input_bytes: StdArc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes,
            vec![input_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_buffer_placeholder_bytes: StdArc<[u8]> =
            vec![0u8; input_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes,
            vec![input_data.len()],
            std::mem::size_of::<f32>(),
        );

        let config_data = vec![0u32];
        let config_bytes: StdArc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let workgroups = [1, 1, 1];
        let result_buffers = cpu
            .dispatch(
                &Kernel::Exp,
                &[input_buffer_view, output_buffer_view, config_buffer_view],
                workgroups,
            )
            .expect("Dispatch for Exp failed");

        assert_eq!(
            result_buffers.len(),
            1,
            "Exp should return one output buffer"
        );
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_output_data.len() * std::mem::size_of::<f32>()
        );

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "Mismatch for Exp. Got: {}, Expected: {}",
                got,
                expected
            );
        }
    }
}
