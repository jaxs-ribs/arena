use crate::{BufferView, ComputeError};

pub fn handle_segmented_reduce_sum(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        return Err(ComputeError::ShapeMismatch(
            "SegmentedReduceSum kernel expects 4 buffers",
        ));
    }
    let data_view = &binds[0];
    let segments_view = &binds[1];
    if data_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "SegmentedReduceSum kernel currently only supports f32 data input",
        ));
    }
    if segments_view.element_size_in_bytes != std::mem::size_of::<u32>() {
        return Err(ComputeError::ShapeMismatch(
            "SegmentedReduceSum kernel currently only supports u32 segment indices",
        ));
    }
    let data_values: &[f32] = bytemuck::cast_slice(&data_view.data);
    let segment_indices: &[u32] = bytemuck::cast_slice(&segments_view.data);

    if segment_indices.is_empty() && !data_values.is_empty() {
        return Err(ComputeError::ShapeMismatch(
            "SegmentedReduceSum received data but no segment indices",
        ));
    }
    if segment_indices.is_empty() && data_values.is_empty() {
        return Ok(vec![Vec::new()]);
    }

    let mut output_sums: Vec<f32> = Vec::with_capacity(segment_indices.len());

    for i in 0..segment_indices.len() {
        let segment_start = segment_indices[i] as usize;
        let segment_end = if i + 1 < segment_indices.len() {
            segment_indices[i + 1] as usize
        } else {
            data_values.len()
        };

        if segment_start > segment_end || segment_end > data_values.len() {
            return Err(ComputeError::ShapeMismatch(
                "Segment indices out of bounds or invalid segment range",
            ));
        }

        let segment_data = &data_values[segment_start..segment_end];
        output_sums.push(segment_data.iter().sum());
    }

    let out_bytes = bytemuck::cast_slice(&output_sums).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_segmented_reduce_sum_computes_segment_sums() {
        let cpu = CpuBackend::new();
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let segment_indices = vec![0u32, 3, 7];
        let expected_sums = vec![6.0f32, 22.0, 27.0];

        let input_bytes: StdArc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes,
            vec![input_data.len()],
            std::mem::size_of::<f32>(),
        );

        let segment_indices_bytes: StdArc<[u8]> =
            bytemuck::cast_slice(&segment_indices).to_vec().into();
        let segment_indices_buffer_view = BufferView::new(
            segment_indices_bytes,
            vec![segment_indices.len()],
            std::mem::size_of::<u32>(),
        );

        let output_placeholder: StdArc<[u8]> =
            vec![0u8; expected_sums.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder,
            vec![expected_sums.len()],
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
            input_buffer_view,
            segment_indices_buffer_view,
            output_buffer_view,
            config_buffer_view,
        ];
        let result_buffers = cpu
            .dispatch(&Kernel::SegmentedReduceSum, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for SegmentedReduceSum failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        assert_eq!(
            output_bytes.len(),
            expected_sums.len() * std::mem::size_of::<f32>()
        );
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_sums.len());
        for (i, (got, expected)) in output_values.iter().zip(expected_sums.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "Mismatch for SegmentedReduceSum at segment index {}. Got: {}, Expected: {}",
                i,
                got,
                expected
            );
        }
    }
}
