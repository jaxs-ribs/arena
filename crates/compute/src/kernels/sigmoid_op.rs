use crate::{BufferView, ComputeError};

pub fn handle_sigmoid(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "Sigmoid kernel expects 3 buffers",
        ));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Sigmoid kernel currently only supports f32 data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
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
    fn test_sigmoid() {
        let cpu = CpuBackend::new();

        let a_data = vec![0.0f32, 1.0, -1.0, 5.0, -5.0];
        let a_bytes = Arc::from(bytemuck::cast_slice(&a_data));
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 5];
        let out_bytes = Arc::from(bytemuck::cast_slice(&out_data));
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes = Arc::from(bytemuck::cast_slice(&config_data));
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![a, out, config];
        let result_buffers = cpu
            .dispatch(&Kernel::Sigmoid, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        let expected = a_data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<f32>>();

        result.iter().zip(expected.iter()).for_each(|(a, b)| {
            assert!((a - b).abs() < 1e-6);
        });
    }
}
