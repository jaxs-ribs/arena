use crate::{kernels, BufferView, ComputeBackend, ComputeError, Kernel};
use std::sync::Arc;

#[derive(Default, Debug, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CpuBackend {
    fn dispatch(
        &self,
        shader: &Kernel,
        binds: &[BufferView],
        _workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        for buffer_view in binds {
            let expected_elements = buffer_view.shape.iter().product::<usize>();
            let expected_bytes = expected_elements * buffer_view.element_size_in_bytes;

            if buffer_view.data.len() != expected_bytes {
                return Err(ComputeError::ShapeMismatch(
                    "Buffer data length does not match product of shape dimensions and element size",
                ));
            }
        }
        let result = match shader {
            Kernel::Add => kernels::handle_add(binds),
            Kernel::Sub => kernels::handle_sub(binds),
            Kernel::Mul => kernels::handle_mul(binds),
            Kernel::Div => kernels::handle_div(binds),
            Kernel::Where => kernels::handle_where(binds),
            Kernel::Neg => kernels::handle_neg(binds),
            Kernel::Exp => kernels::handle_exp(binds),
            Kernel::Log => kernels::handle_log(binds),
            Kernel::Sqrt => kernels::handle_sqrt(binds),
            Kernel::Rsqrt => kernels::handle_rsqrt(binds),
            Kernel::Tanh => kernels::handle_tanh(binds),
            Kernel::Relu => kernels::handle_relu(binds),
            Kernel::Sigmoid => kernels::handle_sigmoid(binds),
            Kernel::Min => kernels::handle_min(binds),
            Kernel::Max => kernels::handle_max(binds),
            Kernel::Clamp => kernels::handle_clamp(binds),
            Kernel::ReduceSum => kernels::handle_reduce_sum(binds),
            Kernel::ReduceMean => kernels::handle_reduce_mean(binds),
            Kernel::ReduceMax => kernels::handle_reduce_max(binds),
            Kernel::SegmentedReduceSum => kernels::handle_segmented_reduce_sum(binds),
            Kernel::ScatterAdd => kernels::handle_scatter_add(binds),
            Kernel::Gather => kernels::handle_gather(binds),
            Kernel::MatMul => kernels::handle_matmul(binds),
            Kernel::IntegrateBodies => kernels::handle_integrate_bodies(binds),
            Kernel::DetectContactsSphere => kernels::handle_detect_contacts_sphere(binds),
            Kernel::DetectContactsBox => kernels::handle_detect_contacts_box(binds),
            Kernel::DetectContactsSDF => kernels::handle_detect_contacts_sdf(binds),
            Kernel::SolveContactsPBD => kernels::handle_solve_contacts_pbd(binds),
            Kernel::SolveJointsPBD => kernels::handle_solve_joints_pbd(binds),
            Kernel::RngNormal => kernels::handle_rng_normal(binds),
            Kernel::ExpandInstances => kernels::handle_expand_instances(binds),
        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_shape_fails() {
        let cpu = CpuBackend::new();
        let bad_buf = BufferView::new(vec![0u8; 12].into(), vec![4], 4);
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8; 4].into(), vec![1], 4);
        let result = cpu.dispatch(&Kernel::Add, &[bad_buf, good_buf, out_buf, cfg], [1, 1, 1]);
        assert!(
            matches!(result, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch error, got {result:?}"
        );
    }
} 