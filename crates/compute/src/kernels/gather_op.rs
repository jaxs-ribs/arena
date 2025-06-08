use crate::{BufferView, ComputeError};

pub fn handle_gather(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        return Err(ComputeError::ShapeMismatch(
            "Gather kernel expects 4 buffers",
        ));
    }
    let source_data_view = &binds[0];
    let indices_view = &binds[1];

    if source_data_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Gather kernel currently only supports f32 source data",
        ));
    }
    if indices_view.element_size_in_bytes != std::mem::size_of::<u32>() {
        return Err(ComputeError::ShapeMismatch(
            "Gather kernel currently only supports u32 for indices",
        ));
    }

    let source_data: &[f32] = bytemuck::cast_slice(&source_data_view.data);
    let indices_to_gather: &[u32] = bytemuck::cast_slice(&indices_view.data);

    if source_data.is_empty() && !indices_to_gather.is_empty() {
        return Err(ComputeError::ShapeMismatch(
            "Gather kernel received indices but no source data",
        ));
    }

    let mut gathered_values: Vec<f32> = Vec::with_capacity(indices_to_gather.len());
    for &index_to_gather_u32 in indices_to_gather {
        let index_to_gather = index_to_gather_u32 as usize;
        if index_to_gather >= source_data.len() {
            return Err(ComputeError::ShapeMismatch(
                "Gather index out of bounds for source data",
            ));
        }
        gathered_values.push(source_data[index_to_gather]);
    }

    let out_bytes = bytemuck::cast_slice(&gathered_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_gather_collects_values_from_indices() {
        let cpu = CpuBackend::new();
        let source_data = vec![10.0f32, 11.0, 12.0, 13.0, 14.0];
        let indices_to_gather = vec![3u32, 0, 2, 2, 4];
        let expected_output_data = vec![13.0f32, 10.0, 12.0, 12.0, 14.0];

        let source_data_bytes: StdArc<[u8]> = bytemuck::cast_slice(&source_data).to_vec().into();
        let source_buffer_view = BufferView::new(
            source_data_bytes,
            vec![source_data.len()],
            std::mem::size_of::<f32>(),
        );

        let indices_bytes: StdArc<[u8]> = bytemuck::cast_slice(&indices_to_gather).to_vec().into();
        let indices_buffer_view = BufferView::new(
            indices_bytes,
            vec![indices_to_gather.len()],
            std::mem::size_of::<u32>(),
        );

        let output_placeholder: StdArc<[u8]> =
            vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder,
            vec![expected_output_data.len()],
            std::mem::size_of::<f32>(),
        );

        let config_data = vec![0u32];
        let config_bytes: StdArc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![config_data.len()],
            std::mem::size_of::<u32>(),
        );

        let dispatch_binds = [
            source_buffer_view,
            indices_buffer_view,
            output_buffer_view,
            config_buffer_view,
        ];
        let result_buffers = cpu
            .dispatch(&Kernel::Gather, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for Gather failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_output_data.len() * std::mem::size_of::<f32>()
        );
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());
        for (i, (got, expected)) in output_values
            .iter()
            .zip(expected_output_data.iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-6,
                "Mismatch for Gather at index {}. Got: {}, Expected: {}",
                i,
                got,
                expected
            );
        }
    }
}
