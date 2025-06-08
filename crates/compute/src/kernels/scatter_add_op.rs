use crate::{BufferView, ComputeError};

pub fn handle_scatter_add(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        return Err(ComputeError::ShapeMismatch(
            "ScatterAdd kernel expects 4 buffers",
        ));
    }
    let values_view = &binds[0];
    let indices_view = &binds[1];
    let accumulator_view = &binds[2];

    if values_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "ScatterAdd kernel currently only supports f32 data for values to add",
        ));
    }
    if indices_view.element_size_in_bytes != std::mem::size_of::<u32>() {
        return Err(ComputeError::ShapeMismatch(
            "ScatterAdd kernel currently only supports u32 for indices",
        ));
    }
    if accumulator_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "ScatterAdd kernel currently only supports f32 for the accumulator buffer",
        ));
    }

    let values_to_add: &[f32] = bytemuck::cast_slice(&values_view.data);
    let indices: &[u32] = bytemuck::cast_slice(&indices_view.data);

    if values_to_add.len() != indices.len() {
        return Err(ComputeError::ShapeMismatch(
            "ScatterAdd requires the number of values to add to match the number of indices",
        ));
    }

    let mut output_accumulator: Vec<f32> =
        bytemuck::cast_slice::<_, f32>(&accumulator_view.data).to_vec();

    for (i, &value_to_add) in values_to_add.iter().enumerate() {
        let scatter_idx = indices[i] as usize;
        if scatter_idx >= output_accumulator.len() {
            return Err(ComputeError::ShapeMismatch(
                "ScatterAdd index out of bounds for the output accumulator buffer",
            ));
        }
        output_accumulator[scatter_idx] += value_to_add;
    }

    let out_bytes = bytemuck::cast_slice(&output_accumulator).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn mock_scatter_add_adds_values_to_indices() {
        let cpu = CpuBackend::new();
        let initial_output_data = vec![0.0f32; 5];
        let values_to_add_data = vec![1.0f32, 2.0, 3.0];
        let indices_data = vec![1u32, 0, 3];
        let expected_final_output_data = vec![2.0f32, 1.0, 0.0, 3.0, 0.0];

        let values_to_add_bytes: Arc<[u8]> =
            bytemuck::cast_slice(&values_to_add_data).to_vec().into();
        let values_buffer_view = BufferView::new(
            values_to_add_bytes,
            vec![values_to_add_data.len()],
            std::mem::size_of::<f32>(),
        );

        let indices_bytes: Arc<[u8]> = bytemuck::cast_slice(&indices_data).to_vec().into();
        let indices_buffer_view = BufferView::new(
            indices_bytes,
            vec![indices_data.len()],
            std::mem::size_of::<u32>(),
        );

        let initial_output_bytes: Arc<[u8]> =
            bytemuck::cast_slice(&initial_output_data).to_vec().into();
        let output_accumulator_buffer_view = BufferView::new(
            initial_output_bytes,
            vec![initial_output_data.len()],
            std::mem::size_of::<f32>(),
        );

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let dispatch_binds = [
            values_buffer_view,
            indices_buffer_view,
            output_accumulator_buffer_view,
            config_buffer_view,
        ];
        let result_buffers = cpu
            .dispatch(&Kernel::ScatterAdd, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for ScatterAdd failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_final_output_data.len() * std::mem::size_of::<f32>()
        );
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_final_output_data.len());
        for (i, (got, expected)) in output_values
            .iter()
            .zip(expected_final_output_data.iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-6,
                "Mismatch for ScatterAdd at index {}. Got: {}, Expected: {}",
                i,
                got,
                expected
            );
        }
    }
}
