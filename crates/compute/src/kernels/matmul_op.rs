use crate::{BufferView, ComputeError};

pub fn handle_matmul(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 4 {
        return Err(ComputeError::ShapeMismatch(
            "MatMul kernel expects 4 buffers",
        ));
    }
    let a_view = &binds[0];
    let b_view = &binds[1];
    let config_view = &binds[3];

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatMulConfig {
        m: u32,
        k: u32,
        n: u32,
    }

    if config_view.data.len() != std::mem::size_of::<MatMulConfig>() {
        return Err(ComputeError::ShapeMismatch(
            "MatMul config buffer has incorrect size",
        ));
    }
    let config: &MatMulConfig = bytemuck::from_bytes(&config_view.data);
    let m = config.m as usize;
    let k = config.k as usize;
    let n = config.n as usize;

    if a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || b_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "MatMul kernel currently only supports f32 data for matrices A and B",
        ));
    }

    let a_data: &[f32] = bytemuck::cast_slice(&a_view.data);
    let b_data: &[f32] = bytemuck::cast_slice(&b_view.data);

    if a_data.len() != m * k {
        return Err(ComputeError::ShapeMismatch(
            "Matrix A data length does not match M*K from config",
        ));
    }
    if b_data.len() != k * n {
        return Err(ComputeError::ShapeMismatch(
            "Matrix B data length does not match K*N from config",
        ));
    }
    if a_view.shape != vec![m, k] {
        return Err(ComputeError::ShapeMismatch(
            "Matrix A shape in BufferView does not match M,K from config",
        ));
    }
    if b_view.shape != vec![k, n] {
        return Err(ComputeError::ShapeMismatch(
            "Matrix B shape in BufferView does not match K,N from config",
        ));
    }

    let mut output_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            output_data[i * n + j] = sum;
        }
    }

    let out_bytes = bytemuck::cast_slice(&output_data).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use crate::{CpuBackend, Kernel, BufferView};

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestMatMulConfig {
        m: u32,
        k: u32,
        n: u32,
    }

    #[test]
    fn mock_matmul_multiplies_matrices() {
        let cpu = CpuBackend::new();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 2u32;
        let k = 3u32;

        let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let n = 2u32;
        let expected_output_data = vec![58.0f32, 64.0, 139.0, 154.0];

        let a_bytes: StdArc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a_buffer_view = BufferView::new(
            a_bytes,
            vec![m as usize, k as usize],
            std::mem::size_of::<f32>(),
        );

        let b_bytes: StdArc<[u8]> = bytemuck::cast_slice(&b_data).to_vec().into();
        let b_buffer_view = BufferView::new(
            b_bytes,
            vec![k as usize, n as usize],
            std::mem::size_of::<f32>(),
        );

        let output_placeholder: StdArc<[u8]> =
            vec![0u8; (m * n) as usize * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder,
            vec![m as usize, n as usize],
            std::mem::size_of::<f32>(),
        );

        let matmul_config = TestMatMulConfig { m, k, n };
        let config_bytes: StdArc<[u8]> = bytemuck::bytes_of(&matmul_config).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes,
            vec![1],
            std::mem::size_of::<TestMatMulConfig>(),
        );

        let dispatch_binds = [
            a_buffer_view,
            b_buffer_view,
            output_buffer_view,
            config_buffer_view,
        ];
        let result_buffers = cpu
            .dispatch(&Kernel::MatMul, &dispatch_binds, [1, 1, 1])
            .expect("Dispatch for MatMul failed");

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
                "Mismatch for MatMul at index {}. Got: {}, Expected: {}",
                i,
                got,
                expected
            );
        }
    }
}
