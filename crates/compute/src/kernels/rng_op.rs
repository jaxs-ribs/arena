#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_rng_normal() {
        let cpu = CpuBackend::new();
        let num_elements = 100usize;
        let out_data = vec![0.0f32; num_elements];
        let out_bytes: Arc<[u8]> = bytemuck::cast_slice(&out_data).to_vec().into();
        let output_placeholder = BufferView::new(out_bytes, vec![num_elements], std::mem::size_of::<f32>());

        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config = BufferView::new(config_bytes, vec![config_data.len()], std::mem::size_of::<u32>());

        let dispatch_binds = vec![output_placeholder, config];
        let result_buffers = cpu
            .dispatch(&Kernel::RngNormal, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result: &[f32] = bytemuck::cast_slice(&result_buffers[0]);
        assert_eq!(result.len(), num_elements);

        // Check that not all values are zero.
        let mut all_zeros = true;
        for &val in result {
            if val != 0.0 {
                all_zeros = false;
                break;
            }
        }
        assert!(!all_zeros, "RNG produced all zeros, which is unlikely.");
    }
} 
