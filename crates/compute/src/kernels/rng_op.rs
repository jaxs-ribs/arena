#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn test_rng_normal() {
        let cpu = CpuBackend::new();
        let num_elements = 100;
        let output_placeholder = BufferView::new(Arc::new(vec![0.0f32; num_elements]), ());

        let dispatch_binds = &[&output_placeholder];
        let result_buffers = cpu
            .dispatch(&Kernel::RngNormal, &dispatch_binds, [1, 1, 1])
            .unwrap();

        let result = result_buffers[0].as_slice::<f32>().unwrap();
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