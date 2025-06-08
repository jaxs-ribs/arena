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

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_where() {
        let cpu = CpuBackend::new();

        let cond = BufferView::from(Arc::new(vec![1u32, 0, 1, 0, 1]));
        let a = BufferView::from(Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        let b = BufferView::from(Arc::new(vec![6.0, 7.0, 8.0, 9.0, 10.0]));
        let out = BufferView::new(Arc::new(vec![0.0; 5]), ());

        let dispatch_binds = &[&cond, &a, &b, &out];
        let result_buffers = cpu
            .dispatch(&Kernel::Where, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
        assert_eq!(result, &[1.0, 7.0, 3.0, 9.0, 5.0]);
    }
}
