use crate::{BufferView, ComputeError};

pub fn handle_reduce_max(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch("ReduceMax kernel expects 3 buffers"));
    }
    let input_view = &binds[0];
    if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
        return Err(ComputeError::ShapeMismatch("ReduceMax kernel currently only supports f32 input data"));
    }
    let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
    let max_value: f32 = input_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let out_bytes = bytemuck::bytes_of(&max_value).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockCpu, ComputeBackend, Kernel};
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_reduce_max_computes_max_value() {
        let cpu = MockCpu::default();
        let input_data1 = vec![1.0f32, -2.0, 5.0, 0.0, 4.5, -10.0];
        let expected_max1: f32 = input_data1.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let input_bytes1: StdArc<[u8]> = bytemuck::cast_slice(&input_data1).to_vec().into();
        let input_buffer_view1 = BufferView::new(input_bytes1, vec![input_data1.len()], std::mem::size_of::<f32>());

        let output_placeholder: StdArc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(output_placeholder.clone(), vec![1], std::mem::size_of::<f32>());
        let config_data = vec![0u32];
        let config_bytes: StdArc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(config_bytes.clone(), vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds1 = [input_buffer_view1, output_buffer_view.clone(), config_buffer_view.clone()];
        let result_buffers1 = cpu.dispatch(&Kernel::ReduceMax, &dispatch_binds1, [1,1,1]).expect("Dispatch for ReduceMax (case 1) failed");

        assert_eq!(result_buffers1.len(), 1);
        let output_bytes1 = &result_buffers1[0];
        assert_eq!(output_bytes1.len(), std::mem::size_of::<f32>());
        let output_value1: f32 = *bytemuck::from_bytes(output_bytes1);
        assert_eq!(output_value1, expected_max1);

        let input_data2: Vec<f32> = Vec::new();
        let expected_max2: f32 = f32::NEG_INFINITY;

        let input_bytes2: StdArc<[u8]> = bytemuck::cast_slice(&input_data2).to_vec().into();
        let input_buffer_view2 = BufferView::new(input_bytes2, vec![input_data2.len()], std::mem::size_of::<f32>());
        let dispatch_binds2 = [input_buffer_view2, output_buffer_view, config_buffer_view];
        let result_buffers2 = cpu.dispatch(&Kernel::ReduceMax, &dispatch_binds2, [1,1,1]).expect("Dispatch for ReduceMax (case 2) failed");

        assert_eq!(result_buffers2.len(), 1);
        let output_bytes2 = &result_buffers2[0];
        assert_eq!(output_bytes2.len(), std::mem::size_of::<f32>());
        let output_value2: f32 = *bytemuck::from_bytes(output_bytes2);
        assert_eq!(output_value2, expected_max2);
    }
}
