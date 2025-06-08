use crate::{BufferView, ComputeError};

pub fn handle_expand_instances(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "ExpandInstances kernel expects 3 buffers",
        ));
    }
    let template_view = &binds[0];
    let config_view = &binds[2];

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct ExpandConfig {
        count: u32,
    }

    if config_view.data.len() != std::mem::size_of::<ExpandConfig>() {
        return Err(ComputeError::ShapeMismatch(
            "ExpandInstances config buffer has incorrect size",
        ));
    }
    let config: &ExpandConfig = bytemuck::from_bytes(&config_view.data);
    let repetition_count = config.count as usize;

    if repetition_count == 0 {
        return Ok(vec![Vec::new()]);
    }

    let template_bytes = &template_view.data;
    let mut output_bytes = Vec::with_capacity(template_bytes.len() * repetition_count);
    for _ in 0..repetition_count {
        output_bytes.extend_from_slice(template_bytes);
    }
    Ok(vec![output_bytes])
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use crate::{ComputeBackend, Kernel, backend::mock_cpu::MockCpu};
    use std::sync::Arc as StdArc;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestExpandConfig {
        count: u32,
    }

    #[test]
    fn mock_expand_instances_repeats_template() {
        let cpu = MockCpu::default();
        let template_data = vec![1.0f32, 2.0, 3.0];
        let repetition_count = 3u32;
        let mut expected_output_data =
            Vec::with_capacity(template_data.len() * repetition_count as usize);
        for _ in 0..repetition_count {
            expected_output_data.extend_from_slice(&template_data);
        }

        let template_bytes: StdArc<[u8]> = bytemuck::cast_slice(&template_data).to_vec().into();
        let template_buffer_view = BufferView::new(
            template_bytes,
            vec![template_data.len()],
            std::mem::size_of::<f32>(),
        );

        let output_placeholder_total_bytes =
            template_data.len() * repetition_count as usize * std::mem::size_of::<f32>();
        let output_placeholder: StdArc<[u8]> = vec![0u8; output_placeholder_total_bytes].into();
        let output_buffer_view = BufferView::new(
            output_placeholder,
            vec![repetition_count as usize, template_data.len()],
            std::mem::size_of::<f32>(),
        );

        let expand_config = TestExpandConfig {
            count: repetition_count,
        };
        let config_bytes: StdArc<[u8]> = bytemuck::bytes_of(&expand_config).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![1],
            std::mem::size_of::<TestExpandConfig>(),
        );

        let dispatch_binds = [template_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu
            .dispatch(&Kernel::ExpandInstances, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for ExpandInstances failed");

        assert_eq!(result_buffers.len(), 1);
        let output_bytes = &result_buffers[0];
        let expected_output_as_bytes: Vec<u8> =
            bytemuck::cast_slice(&expected_output_data).to_vec();
        assert_eq!(output_bytes.len(), expected_output_as_bytes.len());
        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values, expected_output_data.as_slice());
    }
}
