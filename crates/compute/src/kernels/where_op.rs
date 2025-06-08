use crate::{BufferView, ComputeError};

pub fn handle_where(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        // cond, true_val, false_val, out_placeholder per layout.rs
        return Err(ComputeError::ShapeMismatch(
            "Where kernel expects 4 buffers",
        ));
    }
    let cond_view = &binds[0];
    let true_view = &binds[1];
    let false_view = &binds[2];
    // binds[3] is the output placeholder

    if cond_view.element_size_in_bytes != std::mem::size_of::<u32>() {
        return Err(ComputeError::ShapeMismatch(
            "Where kernel condition input currently only supports u32 data",
        ));
    }
    if true_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || false_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "Where kernel true/false inputs currently only support f32 data",
        ));
    }

    let num_cond_elements = cond_view.data.len() / std::mem::size_of::<u32>();
    let num_true_elements = true_view.data.len() / std::mem::size_of::<f32>();
    let num_false_elements = false_view.data.len() / std::mem::size_of::<f32>();

    if !(num_cond_elements == num_true_elements && num_true_elements == num_false_elements) {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Where must have the same number of elements",
        ));
    }
    if !(cond_view.shape == true_view.shape && true_view.shape == false_view.shape) {
        return Err(ComputeError::ShapeMismatch(
            "Input buffers for Where must have the same shape dimensions",
        ));
    }

    let cond_values: &[u32] = bytemuck::cast_slice(&cond_view.data);
    let true_values: &[f32] = bytemuck::cast_slice(&true_view.data);
    let false_values: &[f32] = bytemuck::cast_slice(&false_view.data);

    let output_values: Vec<f32> = cond_values
        .iter()
        .zip(true_values.iter())
        .zip(false_values.iter())
        .map(|((&c, &t), &f)| if c != 0 { t } else { f })
        .collect();

    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use crate::{BufferView, Kernel, CpuBackend};
    use std::sync::Arc;

    #[test]
    fn mock_where_selects_values() {
        let cpu = CpuBackend::new();
        let condition_data = vec![1u32, 0, 1, 0, 1];
        let true_values_data = vec![1.1f32, 2.2, 3.3, 4.4, 5.5];
        let false_values_data = vec![-1.1f32, -2.2, -3.3, -4.4, -5.5];

        let expected_output_data: Vec<f32> = condition_data
            .iter()
            .zip(true_values_data.iter())
            .zip(false_values_data.iter())
            .map(|((&cond, &t_val), &f_val)| if cond != 0 { t_val } else { f_val })
            .collect();

        let condition_bytes: Arc<[u8]> = bytemuck::cast_slice(&condition_data).to_vec().into();
        let condition_buffer_view = BufferView::new(
            condition_bytes,
            vec![condition_data.len()],
            std::mem::size_of::<u32>(),
        );

        let true_values_bytes: Arc<[u8]> =
            bytemuck::cast_slice(&true_values_data).to_vec().into();
        let true_values_buffer_view = BufferView::new(
            true_values_bytes,
            vec![true_values_data.len()],
            std::mem::size_of::<f32>(),
        );

        let false_values_bytes: Arc<[u8]> =
            bytemuck::cast_slice(&false_values_data).to_vec().into();
        let false_values_buffer_view = BufferView::new(
            false_values_bytes,
            vec![false_values_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> =
            vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes,
            vec![expected_output_data.len()],
            std::mem::size_of::<f32>(),
        );

        let workgroups = [1, 1, 1];
        let result_buffers = cpu
            .dispatch(
                &Kernel::Where,
                &[
                    condition_buffer_view,
                    true_values_buffer_view,
                    false_values_buffer_view,
                    output_buffer_view,
                ],
                workgroups,
            )
            .expect("Dispatch for Where failed");

        assert_eq!(
            result_buffers.len(),
            1,
            "Where should return one output buffer"
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
                "Mismatch. Got: {}, Expected: {}",
                got,
                expected
            );
        }
    }
}
