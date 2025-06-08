use crate::{BufferView, ComputeError};

pub fn handle_sqrt(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch("Sqrt kernel expects 3 buffers"));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch(
            "Sqrt kernel currently only supports f32 data",
        ));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let output_values: Vec<f32> = input_values.iter().map(|&x| x.sqrt()).collect();
    let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
    Ok(vec![out_bytes])
}

/*
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_sqrt() {
        let cpu = CpuBackend::new();

        let a = BufferView::from(Arc::new(vec![1.0, 4.0, 9.0, 16.0]));
        let out = BufferView::new(Arc::new(vec![0.0; 4]), ());

        let dispatch_binds = &[&a, &out];
        let result_buffers = cpu
            .dispatch(&Kernel::Sqrt, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }
}
*/
