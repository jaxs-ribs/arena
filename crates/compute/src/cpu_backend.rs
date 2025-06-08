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
            Kernel::Add => kernels::add_op::handle_add(binds),
            Kernel::Sub => kernels::sub_op::handle_sub(binds),
            Kernel::Mul => kernels::mul_op::handle_mul(binds),
            Kernel::Div => kernels::div_op::handle_div(binds),
            Kernel::Where => kernels::where_op::handle_where(binds),
            Kernel::Neg => kernels::neg_op::handle_neg(binds),
            Kernel::Exp => kernels::exp_op::handle_exp(binds),
            Kernel::Log => kernels::log_op::handle_log(binds),
            Kernel::Sqrt => kernels::sqrt_op::handle_sqrt(binds),
            Kernel::Rsqrt => kernels::rsqrt_op::handle_rsqrt(binds),
            Kernel::Tanh => kernels::tanh_op::handle_tanh(binds),
            Kernel::Relu => kernels::relu_op::handle_relu(binds),
            Kernel::Sigmoid => kernels::sigmoid_op::handle_sigmoid(binds),
            Kernel::Min => kernels::min_op::handle_min(binds),
            Kernel::Max => kernels::max_op::handle_max(binds),
            Kernel::Clamp => kernels::clamp_op::handle_clamp(binds),
            Kernel::ReduceSum => kernels::reduce_sum_op::handle_reduce_sum(binds),
            Kernel::ReduceMean => kernels::reduce_mean_op::handle_reduce_mean(binds),
            Kernel::ReduceMax => kernels::reduce_max_op::handle_reduce_max(binds),
            Kernel::SegmentedReduceSum => {
                kernels::segmented_reduce_sum_op::handle_segmented_reduce_sum(binds)
            }
            Kernel::ScatterAdd => kernels::scatter_add_op::handle_scatter_add(binds),
            Kernel::Gather => kernels::gather_op::handle_gather(binds),
            Kernel::MatMul => kernels::matmul_op::handle_matmul(binds),
            Kernel::IntegrateBodies => kernels::integrate_bodies_op::handle_integrate_bodies(binds),
            Kernel::DetectContactsSphere => {
                kernels::detect_contacts_sphere::handle_detect_contacts_sphere(binds)
            }
            Kernel::DetectContactsBox => {
                kernels::detect_contacts_box_op::handle_detect_contacts_box(binds)
            }
            Kernel::DetectContactsSDF => {
                kernels::detect_contacts_sdf_op::handle_detect_contacts_sdf(binds)
            }
            Kernel::SolveContactsPBD => {
                kernels::solve_contacts_pbd_op::handle_solve_contacts_pbd(binds)
            }
            Kernel::SolveJointsPBD => kernels::solve_joints_pbd_op::handle_solve_joints_pbd(binds),
            Kernel::RngNormal => kernels::rng_normal_op::handle_rng_normal(binds),
            Kernel::ExpandInstances => kernels::expand_instances_op::handle_expand_instances(binds),
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