use crate::{BufferView, ComputeError};

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

        let a = BufferView::from(Arc::new(vec![1.0, -1.0, 0.0, 5.0, -5.0]));
        let out = BufferView::new(Arc::new(vec![0.0; 5]), ());

        let dispatch_binds = &[&a, &out];
        let result_buffers = cpu
            .dispatch(&Kernel::Neg, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
        assert_eq!(result, &[-1.0, 1.0, 0.0, -5.0, 5.0]);
    }
}
