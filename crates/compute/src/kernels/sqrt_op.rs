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

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_sqrt() {
        let cpu = CpuBackend::new();

        let a_data = vec![4.0f32, 9.0, 0.0, 16.0];
        let a_bytes = Arc::from(bytemuck::cast_slice(&a_data));
        let a = BufferView::new(a_bytes, vec![a_data.len()], std::mem::size_of::<f32>());

        let out_data = vec![0.0f32; 4];
        let out_bytes = Arc::from(bytemuck::cast_slice(&out_data));
        let out = BufferView::new(out_bytes, vec![out_data.len()], std::mem::size_of::<f32>());

        let dispatch_binds = vec![a, out];
        let result_buffers = cpu
            .dispatch(&Kernel::Sqrt, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result, &[2.0, 3.0, 0.0, 4.0]);
    }
}
