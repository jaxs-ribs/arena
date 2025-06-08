// This module re-exports handlers for each kernel operation.

// Element-wise operations
pub mod add_op;
pub use add_op::handle_add;
pub mod sub_op;
pub use sub_op::handle_sub;
pub mod mul_op;
pub use mul_op::handle_mul;
pub mod div_op;
pub use div_op::handle_div;
pub mod neg_op;
pub use neg_op::handle_neg;
pub mod exp_op;
pub use exp_op::handle_exp;
pub mod log_op;
pub use log_op::handle_log;
pub mod sqrt_op;
pub use sqrt_op::handle_sqrt;
pub mod rsqrt_op;
pub use rsqrt_op::handle_rsqrt;
pub mod tanh_op;
pub use tanh_op::handle_tanh;
pub mod relu_op;
pub use relu_op::handle_relu;
pub mod sigmoid_op;
pub use sigmoid_op::handle_sigmoid;
pub mod min_op;
pub use min_op::handle_min;
pub mod max_op;
pub use max_op::handle_max;
pub mod clamp_op;
pub use clamp_op::handle_clamp;
pub mod where_op;
pub use where_op::handle_where;

// Reductions and scatters
pub mod reduce_sum_op;
pub use reduce_sum_op::handle_reduce_sum;
pub mod reduce_mean_op;
pub use reduce_mean_op::handle_reduce_mean;
pub mod reduce_max_op;
pub use reduce_max_op::handle_reduce_max;
pub mod segmented_reduce_sum_op;
pub use segmented_reduce_sum_op::handle_segmented_reduce_sum;
pub mod scatter_add_op;
pub use scatter_add_op::handle_scatter_add;
pub mod gather_op;
pub use gather_op::handle_gather;

// Linear algebra
pub mod matmul_op;
pub use matmul_op::handle_matmul;

// Physics
pub mod integrate_bodies_op;
pub use integrate_bodies_op::handle_integrate_bodies;
pub mod detect_contacts_sphere;
pub use detect_contacts_sphere::handle_detect_contacts_sphere;
pub mod detect_contacts_box_op;
pub use detect_contacts_box_op::handle_detect_contacts_box;
pub mod detect_contacts_sdf_op;
pub use detect_contacts_sdf_op::handle_detect_contacts_sdf;
pub mod solve_contacts_pbd_op;
pub use solve_contacts_pbd_op::handle_solve_contacts_pbd;
pub mod solve_joints_pbd_op;
pub use solve_joints_pbd_op::handle_solve_joints_pbd;

// Misc
pub mod expand_instances_op;
pub use expand_instances_op::handle_expand_instances;
pub mod rng_normal_op;
pub use rng_normal_op::handle_rng_normal;
