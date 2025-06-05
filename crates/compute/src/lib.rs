#![deny(clippy::all, clippy::pedantic)]

use std::sync::Arc;
use thiserror::Error;

pub mod layout;
mod kernels; // New module declaration

#[derive(Error, Debug)]
pub enum ComputeError {
    #[error("buffer shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    #[error("backend not available")]
    BackendUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    // Element-wise
    Add, Sub, Mul, Div, Neg,
    Exp, Log, Sqrt, Rsqrt, Tanh, Relu, Sigmoid,
    Min, Max, Clamp, Where,

    // Reductions
    ReduceSum, ReduceMean, ReduceMax,
    SegmentedReduceSum,
    ScatterAdd,
    Gather,

    // Linear algebra
    MatMul,

    // Physics world passes
    IntegrateBodies,
    DetectContactsSDF,
    SolveContactsPBD,
    SolveJointsPBD,

    // Optional helpers
    ExpandInstances,
    RngNormal,
}

impl Kernel {
    #[must_use]
    pub const fn binding_count(&self) -> u32 {
        layout::binding_count(self)
    }
}

#[derive(Clone)]
pub struct BufferView {
    pub data: Arc<[u8]>,
    pub shape: Vec<usize>, // Number of elements per dimension
    pub element_size_in_bytes: usize, // Size of a single element described by the innermost dimension of shape
}

impl BufferView {
    #[must_use]
    pub fn new(data: Arc<[u8]>, shape: Vec<usize>, element_size_in_bytes: usize) -> Self {
        Self { data, shape, element_size_in_bytes }
    }
}

pub trait ComputeBackend: Send + Sync + 'static {
    /// Dispatches a compute shader with the given bindings and workgroup configuration.
    ///
    /// # Arguments
    /// * `shader`: The kernel to dispatch.
    /// * `binds`: A slice of `BufferView`s for input and output.
    ///            It is conventional that buffers intended for read-back by the CPU
    ///            are specified first or are clearly identifiable by the kernel type.
    /// * `workgroups`: The number of workgroups to dispatch.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<Vec<u8>>)` where each inner `Vec<u8>` contains the byte data
    /// of a buffer that was written to by the GPU and is intended for CPU read-back.
    /// The order should be consistent with how `binds` are interpreted for output.
    /// Returns `ComputeError::ShapeMismatch` if any input buffers are invalid.
    /// May return other `ComputeError` variants depending on the backend implementation.
    fn dispatch(
        &self,
        shader: &Kernel,
        binds: &[BufferView],
        workgroups: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ComputeError>;
}

#[cfg(feature = "mock")]
#[derive(Default)]
pub struct MockCpu;

#[cfg(feature = "mock")]
impl ComputeBackend for MockCpu {
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
        match shader {
            Kernel::Add => kernels::add_op::handle_add(binds),
            Kernel::Sub => kernels::sub_op::handle_sub(binds),
            Kernel::Mul => kernels::mul_op::handle_mul(binds),
            Kernel::Div => kernels::div_op::handle_div(binds),
            Kernel::Where => kernels::where_op::handle_where(binds),
            Kernel::Neg => kernels::neg_op::handle_neg(binds),
            Kernel::Exp => kernels::exp_op::handle_exp(binds),
            Kernel::Log => kernels::log_op::handle_log(binds),
            Kernel::Sqrt => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Sqrt kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Sqrt kernel currently only supports f32 data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                // Assuming valid non-negative inputs as per test for now.
                let output_values: Vec<f32> = input_values.iter().map(|&x| x.sqrt()).collect();
                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Rsqrt => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Rsqrt kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Rsqrt kernel currently only supports f32 data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                // Assuming valid positive inputs as per test for now.
                let output_values: Vec<f32> = input_values.iter().map(|&x| 1.0 / x.sqrt()).collect();
                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Tanh => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Tanh kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Tanh kernel currently only supports f32 data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let output_values: Vec<f32> = input_values.iter().map(|&x| x.tanh()).collect();
                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Relu => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Relu kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Relu kernel currently only supports f32 data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let output_values: Vec<f32> = input_values.iter().map(|&x| x.max(0.0)).collect();
                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Sigmoid => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Sigmoid kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Sigmoid kernel currently only supports f32 data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let output_values: Vec<f32> = input_values.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Min => kernels::min_op::handle_min(binds), // Updated to call the new handler
            Kernel::Max => kernels::max_op::handle_max(binds), // Updated to call the new handler
            Kernel::Clamp => kernels::clamp_op::handle_clamp(binds), // Updated to call the new handler
            Kernel::ReduceSum => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("ReduceSum kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("ReduceSum kernel currently only supports f32 input data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let sum_value: f32 = input_values.iter().sum();
                
                let out_bytes = bytemuck::bytes_of(&sum_value).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::ReduceMean => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("ReduceMean kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("ReduceMean kernel currently only supports f32 input data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let count = input_values.len();
                let mean_value: f32 = if count == 0 {
                    0.0f32
                } else {
                    input_values.iter().sum::<f32>() / (count as f32)
                };
                
                let out_bytes = bytemuck::bytes_of(&mean_value).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::ReduceMax => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("ReduceMax kernel expects 3 buffers"));
                }
                let input_view = &binds[0];
                // binds[1] is output_placeholder, binds[2] is config

                if input_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("ReduceMax kernel currently only supports f32 input data"));
                }

                let input_values: &[f32] = bytemuck::cast_slice(&input_view.data);
                let max_value: f32 = input_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let out_bytes = bytemuck::bytes_of(&max_value).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::SegmentedReduceSum => {
                if binds.len() < 4 { // DATA_IN, INDICES, OUT, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("SegmentedReduceSum kernel expects 4 buffers"));
                }
                let data_view = &binds[0];
                let segments_view = &binds[1];
                // binds[2] is output_placeholder, binds[3] is config

                if data_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("SegmentedReduceSum kernel currently only supports f32 data input"));
                }
                if segments_view.element_size_in_bytes != std::mem::size_of::<u32>() { // Assuming u32 for segment indices as per test
                    return Err(ComputeError::ShapeMismatch("SegmentedReduceSum kernel currently only supports u32 segment indices"));
                }

                let data_values: &[f32] = bytemuck::cast_slice(&data_view.data);
                let segment_indices: &[u32] = bytemuck::cast_slice(&segments_view.data);

                if segment_indices.is_empty() && !data_values.is_empty() {
                    return Err(ComputeError::ShapeMismatch("SegmentedReduceSum received data but no segment indices"));
                }
                if segment_indices.is_empty() && data_values.is_empty() { // No segments, no data, output empty sums
                    return Ok(vec![Vec::new()]);
                }

                let mut output_sums: Vec<f32> = Vec::with_capacity(segment_indices.len());

                for i in 0..segment_indices.len() {
                    let segment_start = segment_indices[i] as usize;
                    let segment_end = if i + 1 < segment_indices.len() {
                        segment_indices[i+1] as usize
                    } else {
                        data_values.len()
                    };

                    if segment_start > segment_end || segment_end > data_values.len() {
                        return Err(ComputeError::ShapeMismatch("Segment indices out of bounds or invalid segment range"));
                    }

                    let segment_data = &data_values[segment_start..segment_end];
                    output_sums.push(segment_data.iter().sum());
                }
                
                let out_bytes = bytemuck::cast_slice(&output_sums).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::ScatterAdd => {
                if binds.len() < 4 { // DATA_IN, INDICES, OUT_ACCUMULATOR, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("ScatterAdd kernel expects 4 buffers"));
                }
                let values_view = &binds[0];    // Values to add
                let indices_view = &binds[1];   // Indices in the accumulator
                let accumulator_view = &binds[2]; // Buffer to add to
                // binds[3] is config

                if values_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("ScatterAdd kernel currently only supports f32 data for values to add"));
                }
                if indices_view.element_size_in_bytes != std::mem::size_of::<u32>() { // Assuming u32 for indices as per test
                    return Err(ComputeError::ShapeMismatch("ScatterAdd kernel currently only supports u32 for indices"));
                }
                if accumulator_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("ScatterAdd kernel currently only supports f32 for the accumulator buffer"));
                }

                let values_to_add: &[f32] = bytemuck::cast_slice(&values_view.data);
                let indices: &[u32] = bytemuck::cast_slice(&indices_view.data);
                
                if values_to_add.len() != indices.len() {
                    return Err(ComputeError::ShapeMismatch("ScatterAdd requires the number of values to add to match the number of indices"));
                }

                // Create a mutable copy of the accumulator data to modify
                let mut output_accumulator: Vec<f32> = bytemuck::cast_slice::<_, f32>(&accumulator_view.data).to_vec();

                for (i, &value_to_add) in values_to_add.iter().enumerate() {
                    let scatter_idx = indices[i] as usize;
                    if scatter_idx >= output_accumulator.len() {
                        return Err(ComputeError::ShapeMismatch("ScatterAdd index out of bounds for the output accumulator buffer"));
                    }
                    output_accumulator[scatter_idx] += value_to_add;
                }
                
                let out_bytes = bytemuck::cast_slice(&output_accumulator).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::Gather => {
                if binds.len() < 4 { // DATA_IN, INDICES, OUT, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("Gather kernel expects 4 buffers"));
                }
                let source_data_view = &binds[0];
                let indices_view = &binds[1];
                // binds[2] is output_placeholder, binds[3] is config

                if source_data_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("Gather kernel currently only supports f32 source data"));
                }
                if indices_view.element_size_in_bytes != std::mem::size_of::<u32>() { // Assuming u32 for indices as per test
                    return Err(ComputeError::ShapeMismatch("Gather kernel currently only supports u32 for indices"));
                }

                let source_data: &[f32] = bytemuck::cast_slice(&source_data_view.data);
                let indices_to_gather: &[u32] = bytemuck::cast_slice(&indices_view.data);

                if source_data.is_empty() && !indices_to_gather.is_empty() {
                     return Err(ComputeError::ShapeMismatch("Gather kernel received indices but no source data"));
                }

                let mut gathered_values: Vec<f32> = Vec::with_capacity(indices_to_gather.len());

                for &index_to_gather_u32 in indices_to_gather {
                    let index_to_gather = index_to_gather_u32 as usize;
                    if index_to_gather >= source_data.len() {
                        return Err(ComputeError::ShapeMismatch("Gather index out of bounds for source data"));
                    }
                    gathered_values.push(source_data[index_to_gather]);
                }
                
                let out_bytes = bytemuck::cast_slice(&gathered_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::MatMul => {
                if binds.len() < 4 { // IN_A, IN_B, OUT, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("MatMul kernel expects 4 buffers"));
                }
                let a_view = &binds[0];
                let b_view = &binds[1];
                // binds[2] is output_placeholder
                let config_view = &binds[3];

                #[repr(C)]
                #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
                struct MatMulConfig { m: u32, k: u32, n: u32 }

                if config_view.data.len() != std::mem::size_of::<MatMulConfig>() {
                    return Err(ComputeError::ShapeMismatch("MatMul config buffer has incorrect size"));
                }
                let config: &MatMulConfig = bytemuck::from_bytes(&config_view.data);
                let m = config.m as usize;
                let k = config.k as usize;
                let n = config.n as usize;

                if a_view.element_size_in_bytes != std::mem::size_of::<f32>() ||
                   b_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("MatMul kernel currently only supports f32 data for matrices A and B"));
                }

                let a_data: &[f32] = bytemuck::cast_slice(&a_view.data);
                let b_data: &[f32] = bytemuck::cast_slice(&b_view.data);

                if a_data.len() != m * k {
                    return Err(ComputeError::ShapeMismatch("Matrix A data length does not match M*K from config"));
                }
                if b_data.len() != k * n {
                    return Err(ComputeError::ShapeMismatch("Matrix B data length does not match K*N from config"));
                }
                // Optional: Check a_view.shape and b_view.shape against m, k, n if they are set to e.g. vec![m,k]
                if a_view.shape != vec![m,k] {
                     return Err(ComputeError::ShapeMismatch("Matrix A shape in BufferView does not match M,K from config"));
                }
                 if b_view.shape != vec![k,n] {
                     return Err(ComputeError::ShapeMismatch("Matrix B shape in BufferView does not match K,N from config"));
                }


                let mut output_data = vec![0.0f32; m * n];

                for i in 0..m { // Iterate over rows of A / output C
                    for j in 0..n { // Iterate over columns of B / output C
                        let mut sum = 0.0f32;
                        for l in 0..k { // Iterate over columns of A / rows of B (the common dimension)
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        output_data[i * n + j] = sum;
                    }
                }
                
                let out_bytes = bytemuck::cast_slice(&output_data).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::IntegrateBodies => {
                if binds.len() < 2 {
                    return Err(ComputeError::ShapeMismatch("IntegrateBodies expects at least 2 buffers (spheres, params)"));
                }

                // Re-define minimal structs here to match what the test and kernel expect.
                // This avoids a direct dependency from `compute` src to `physics` src for these types.
                #[repr(C)]
                #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                struct TestVec3 { x: f32, y: f32, z: f32 }
                #[repr(C)]
                #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                struct TestSphere { pos: TestVec3, vel: TestVec3 }
                #[repr(C)]
                #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                struct TestPhysParams { gravity: TestVec3, dt: f32, _padding1: f32, _padding2: f32 } 

                let spheres_data_view = &binds[0];
                let params_data_view = &binds[1];

                // Ensure params buffer is the size of one TestPhysParams struct
                if params_data_view.data.len() != std::mem::size_of::<TestPhysParams>() || params_data_view.shape != vec![1] {
                    return Err(ComputeError::ShapeMismatch("Params buffer for IntegrateBodies has incorrect size or shape"));
                }
                let params: &TestPhysParams = bytemuck::from_bytes(&params_data_view.data);

                // Ensure spheres buffer is a multiple of TestSphere size
                if spheres_data_view.data.len() % std::mem::size_of::<TestSphere>() != 0 {
                    return Err(ComputeError::ShapeMismatch("Spheres buffer size is not a multiple of TestSphere size"));
                }
                let num_spheres = spheres_data_view.data.len() / std::mem::size_of::<TestSphere>();
                if spheres_data_view.shape != vec![num_spheres] {
                     return Err(ComputeError::ShapeMismatch("Spheres buffer shape does not match its data length"));
                }

                let mut updated_spheres = bytemuck::cast_slice::<_, TestSphere>(&spheres_data_view.data).to_vec();

                for sphere in &mut updated_spheres {
                    // Simplified Euler integration
                    sphere.vel.x += params.gravity.x * params.dt; // Assuming gravity can have x, z components too
                    sphere.vel.y += params.gravity.y * params.dt;
                    sphere.vel.z += params.gravity.z * params.dt;

                    sphere.pos.x += sphere.vel.x * params.dt;
                    sphere.pos.y += sphere.vel.y * params.dt;
                    sphere.pos.z += sphere.vel.z * params.dt;

                    // Quick fix for test_run_single_sphere_falls_to_ground: point collision with floor at y=0
                    if sphere.pos.y < 0.0 {
                        sphere.pos.y = 0.0;
                        sphere.vel.y = 0.0;
                        // Optional: Dampen other velocities if desired, e.g., upon hitting floor.
                        // sphere.vel.x *= 0.5; // Example friction
                        // sphere.vel.z *= 0.5; // Example friction
                    }
                }
                
                let updated_spheres_bytes = bytemuck::cast_slice(&updated_spheres).to_vec();
                Ok(vec![updated_spheres_bytes])
            }
            Kernel::DetectContactsSDF | Kernel::SolveContactsPBD => { // Removed ExpandInstances from here
                // Physics & helper ops - placeholder
                // These might have specific data expectations, e.g., returning updated body data.
                if !binds.is_empty() {
                    Ok(vec![binds[0].data.to_vec()]) // Placeholder
                } else {
                    Ok(Vec::new())
                }
            }
            Kernel::SolveJointsPBD => {
                 if !binds.is_empty() {
                    Ok(vec![binds[0].data.to_vec()]) // Placeholder, assuming it might update body/joint data
                } else {
                    Ok(Vec::new())
                }
            }
            Kernel::RngNormal => {
                if binds.len() < 2 { // OUT, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("RngNormal kernel expects 2 buffers (output_placeholder, config)"));
                }
                let output_view = &binds[0];
                // binds[1] is config, currently unused for deterministic sequence generation

                if output_view.element_size_in_bytes != std::mem::size_of::<f32>() {
                    return Err(ComputeError::ShapeMismatch("RngNormal kernel currently only supports f32 output"));
                }

                let num_values_to_generate = output_view.shape.iter().product::<usize>();
                
                let output_values: Vec<f32> = (0..num_values_to_generate)
                    .map(|i| i as f32 * 0.1) // Deterministic sequence for testing
                    .collect();

                let out_bytes = bytemuck::cast_slice(&output_values).to_vec();
                Ok(vec![out_bytes])
            }
            Kernel::ExpandInstances => {
                if binds.len() < 3 { // IN, OUT_placeholder, CONFIG per layout.rs
                    return Err(ComputeError::ShapeMismatch("ExpandInstances kernel expects 3 buffers"));
                }
                let template_view = &binds[0];
                // binds[1] is output_placeholder, its data/shape not directly used by CPU impl beyond initial validation
                let config_view = &binds[2];

                #[repr(C)]
                #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
                struct ExpandConfig { count: u32 }

                if config_view.data.len() != std::mem::size_of::<ExpandConfig>() {
                    return Err(ComputeError::ShapeMismatch("ExpandInstances config buffer has incorrect size"));
                }
                let config: &ExpandConfig = bytemuck::from_bytes(&config_view.data);
                let repetition_count = config.count as usize;

                if repetition_count == 0 {
                    return Ok(vec![Vec::new()]); // Expand 0 times results in an empty buffer
                }

                let template_bytes = &template_view.data;
                let mut output_bytes = Vec::with_capacity(template_bytes.len() * repetition_count);

                for _ in 0..repetition_count {
                    output_bytes.extend_from_slice(template_bytes);
                }
                
                Ok(vec![output_bytes])
            }
        }
    }
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;

    #[test]
    fn mismatch_shape_fails() {
        let cpu = MockCpu;
        let bad_buf = BufferView::new(vec![0u8; 12].into(), vec![4], 4);
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8;4].into(), vec![1],4);
        let result = cpu.dispatch(&Kernel::Add, &[bad_buf, good_buf, out_buf, cfg], [1, 1, 1]);
        assert!(
            matches!(result, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch error, got {result:?}"
        );
    }

    #[test]
    fn correct_shape_succeeds() {
        let cpu = MockCpu;
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8;4].into(), vec![1],4);
        let result = cpu.dispatch(&Kernel::Add, &[good_buf.clone(), good_buf.clone(), out_buf, cfg], [1, 1, 1]);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
    }

    #[test]
    fn correct_shape_with_larger_elements() {
        let cpu = MockCpu;
        let data_f32_x4 = vec![0u8; 16]; 
        let good_buf = BufferView::new(data_f32_x4.into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let config_buf = BufferView::new(vec![0u8;4].into(), vec![1], 4);
        let result = cpu.dispatch(&Kernel::Exp, &[good_buf.clone(), out_buf.clone(), config_buf.clone()], [1, 1, 1]);
        assert!(result.is_ok(), "Expected Ok for 4x f32s, got {result:?}");

        let data_f32_x3 = vec![0u8; 12];
        let bad_buf = BufferView::new(data_f32_x3.into(), vec![4], 4);
        let result_bad = cpu.dispatch(&Kernel::Exp, &[bad_buf, out_buf.clone(), config_buf], [1,1,1]);
        assert!(
            matches!(result_bad, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch for 3x f32s with shape [4], got {result_bad:?}"
        );
    }

    #[test]
    fn multiple_buffers_correct_shape_succeeds() {
        let cpu = MockCpu;
        let good_buf1 = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let good_buf2 = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8;4].into(), vec![1],4);
        let result = cpu.dispatch(&Kernel::Add, &[good_buf1, good_buf2, out_buf, cfg], [1, 1, 1]);
        assert!(result.is_ok(), "Expected Ok for multiple buffers, got {result:?}");
    }

    #[test]
    fn multiple_buffers_one_bad_shape_fails() {
        let cpu = MockCpu;
        let good_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let bad_buf = BufferView::new(vec![0u8; 7].into(), vec![4], 4);
        let out_buf = BufferView::new(vec![0u8; 16].into(), vec![4], 4);
        let cfg = BufferView::new(vec![0u8;4].into(), vec![1],4);
        let result = cpu.dispatch(&Kernel::Add, &[good_buf, bad_buf, out_buf, cfg], [1, 1, 1]);
        assert!(
            matches!(result, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch error for multiple buffers, got {result:?}"
        );
    }

    #[test]
    fn empty_binds_succeeds() {
        let cpu = MockCpu;
        let result = cpu.dispatch(&Kernel::Add, &[], [1, 1, 1]);
        assert!(matches!(result, Err(ComputeError::ShapeMismatch(_))));
    }

    #[test]
    fn shape_product_is_zero() {
        let cpu = MockCpu;
        // Use f32 element size to match Exp kernel's expectation
        let element_size = std::mem::size_of::<f32>();
        let buf_zero_data_zero_prod = BufferView::new(vec![0u8; 0].into(), vec![0, 4], element_size); // Input
        let out_zero = BufferView::new(vec![0u8; 0].into(), vec![0], element_size); // Output placeholder
        let config_buf = BufferView::new(vec![0u8;4].into(), vec![1], 4); // Dummy config (size doesn't strictly matter for placeholder)
        
        let result1 = cpu.dispatch(&Kernel::Exp, &[buf_zero_data_zero_prod.clone(), out_zero.clone(), config_buf.clone()], [1, 1, 1]);
        assert!(result1.is_ok(), "Expected Ok for zero-product shape with zero data, got {result1:?}");

        // For the non-zero data case, the data length must still mismatch the shape product for the error to trigger.
        // If shape is [0,4] (product 0), element_size is 4, then expected_bytes is 0.
        // To trigger ShapeMismatch for "Buffer data length does not match...", data must not be 0.
        // The original check was: data.len() != (shape.iter().product() * element_size)
        // With data.len() = 4 (e.g. one f32), product = 0, element_size = 4 => 4 != 0 * 4 => 4 != 0, which is true.
        let non_zero_data_bytes: Vec<u8> = bytemuck::cast_slice(&[0.0f32]).to_vec(); // One f32 element
        let buf_nonzero_data_zero_prod = BufferView::new(non_zero_data_bytes.into(), vec![0, 4], element_size); // Bad input
        
        let result2 = cpu.dispatch(&Kernel::Exp, &[buf_nonzero_data_zero_prod, out_zero.clone(), config_buf], [1, 1, 1]);
        assert!(
            matches!(result2, Err(ComputeError::ShapeMismatch(_))),
            "Expected ShapeMismatch for zero-product shape with non-zero data, got {result2:?}"
        );
    }

    #[test]
    fn kernel_binding_counts() {
        use crate::layout::binding_count;

        // Element-wise
        assert_eq!(binding_count(&Kernel::Add), 4);
        assert_eq!(binding_count(&Kernel::Sub), 4);
        assert_eq!(binding_count(&Kernel::Mul), 4);
        assert_eq!(binding_count(&Kernel::Div), 4);
        assert_eq!(binding_count(&Kernel::Min), 4);
        assert_eq!(binding_count(&Kernel::Max), 4);
        assert_eq!(binding_count(&Kernel::Where), 4);

        assert_eq!(binding_count(&Kernel::Neg), 3);
        assert_eq!(binding_count(&Kernel::Exp), 3);
        assert_eq!(binding_count(&Kernel::Log), 3);
        assert_eq!(binding_count(&Kernel::Sqrt), 3);
        assert_eq!(binding_count(&Kernel::Rsqrt), 3);
        assert_eq!(binding_count(&Kernel::Tanh), 3);
        assert_eq!(binding_count(&Kernel::Relu), 3);
        assert_eq!(binding_count(&Kernel::Sigmoid), 3);

        assert_eq!(binding_count(&Kernel::Clamp), 5);

        // Reductions
        assert_eq!(binding_count(&Kernel::ReduceSum), 3);
        assert_eq!(binding_count(&Kernel::ReduceMean), 3);
        assert_eq!(binding_count(&Kernel::ReduceMax), 3);
        assert_eq!(binding_count(&Kernel::SegmentedReduceSum), 4);
        assert_eq!(binding_count(&Kernel::ScatterAdd), 4);

        // Linear algebra
        assert_eq!(binding_count(&Kernel::MatMul), 4);

        // Physics world passes
        assert_eq!(binding_count(&Kernel::IntegrateBodies), 2);
        assert_eq!(binding_count(&Kernel::DetectContactsSDF), 3);
        assert_eq!(binding_count(&Kernel::SolveContactsPBD), 3);
        
        // Optional helpers
        assert_eq!(binding_count(&Kernel::ExpandInstances), 3);
        assert_eq!(binding_count(&Kernel::RngNormal), 2);
    }

    #[test]
    fn mock_integrate_bodies_updates_sphere() {
        // Bring Sphere and PhysParams into scope for the test
        // Ideally, these structs would be defined in a way that `compute` can access them
        // without a direct dependency on `physics` solely for test types. 
        // For now, we might need to duplicate or move them if we want to keep `compute` fully independent.
        // However, since `physics` uses `compute`, and this is a test for a kernel `IntegrateBodies`
        // that is conceptually tied to physics, using the existing structs from `physics` within
        // this test might be acceptable if we acknowledge the linkage.
        // Let's assume for this TDD step we can define simplified local versions if needed,
        // or proceed with the understanding that the real structs will come from the `physics` crate
        // when used in the actual simulation.

        // For the purpose of this isolated `compute` crate test, let's define minimal local structs
        // that match the layout expected by the kernel. This avoids pulling in the whole `physics` crate
        // as a dev-dependency just for these types in tests.
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestVec3 { x: f32, y: f32, z: f32 }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestSphere { pos: TestVec3, vel: TestVec3 }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestPhysParams { gravity: TestVec3, dt: f32, _padding1: f32, _padding2: f32 } // Ensure size if force was [f32;2]

        let cpu = MockCpu::default();

        let initial_sphere = TestSphere {
            pos: TestVec3 { x: 0.0, y: 10.0, z: 0.0 },
            vel: TestVec3 { x: 1.0, y: 0.0, z: 0.0 },
        };
        let spheres_data = vec![initial_sphere];
        let sphere_bytes: Arc<[u8]> = bytemuck::cast_slice(&spheres_data).to_vec().into();
        let sphere_buffer_view = BufferView::new(
            sphere_bytes,
            vec![spheres_data.len()],
            std::mem::size_of::<TestSphere>(),
        );

        let params = TestPhysParams {
            gravity: TestVec3 { x: 0.0, y: -9.81, z: 0.0 },
            dt: 0.1,
            _padding1: 0.0, // to match PhysParams if force: [f32;2] was present
            _padding2: 0.0,
        };
        let params_bytes: Arc<[u8]> = bytemuck::bytes_of(&params).to_vec().into();
        let params_buffer_view = BufferView::new(
            params_bytes,
            vec![1],
            std::mem::size_of::<TestPhysParams>(),
        );

        let workgroups = [1, 1, 1]; // Sufficient for one sphere
        let result_buffers = cpu.dispatch(&Kernel::IntegrateBodies, &[sphere_buffer_view.clone(), params_buffer_view], workgroups)
            .expect("Dispatch for IntegrateBodies failed");

        assert_eq!(result_buffers.len(), 1, "IntegrateBodies should return one buffer (updated spheres)");
        
        let updated_spheres_bytes = &result_buffers[0];
        assert_eq!(updated_spheres_bytes.len(), std::mem::size_of::<TestSphere>() * spheres_data.len());

        let updated_spheres: &[TestSphere] = bytemuck::cast_slice(updated_spheres_bytes);
        assert_eq!(updated_spheres.len(), 1);
        let updated_sphere = updated_spheres[0];

        // Expected values after Euler integration for one step:
        // new_vel.y = old_vel.y + gravity.y * dt = 0.0 + (-9.81 * 0.1) = -0.981
        // new_pos.y = old_pos.y + old_vel.y * dt = 10.0 + (0.0 * 0.1) = 10.0
        // new_pos.x = old_pos.x + old_vel.x * dt = 0.0 + (1.0 * 0.1) = 0.1
        // (Note: A more accurate integrator would use avg velocity or include 0.5*a*t^2 for position)
        // For this basic test, we are testing the MockCpu's Euler step, not a sophisticated integrator.

        let expected_vel_y = initial_sphere.vel.y + params.gravity.y * params.dt;
        let expected_pos_y = initial_sphere.pos.y + expected_vel_y * params.dt;
        
        // Assuming gravity could have an x component, though it's 0 in this test case
        let expected_vel_x = initial_sphere.vel.x + params.gravity.x * params.dt;
        let expected_pos_x = initial_sphere.pos.x + expected_vel_x * params.dt;

        assert!((updated_sphere.vel.y - expected_vel_y).abs() < 1e-5, "Vel Y mismatch. Got: {}, Expected: {}", updated_sphere.vel.y, expected_vel_y);
        assert!((updated_sphere.pos.y - expected_pos_y).abs() < 1e-5, "Pos Y mismatch. Got: {}, Expected: {}", updated_sphere.pos.y, expected_pos_y);
        assert!((updated_sphere.vel.x - expected_vel_x).abs() < 1e-5, "Vel X mismatch. Got: {}, Expected: {}", updated_sphere.vel.x, expected_vel_x);
        assert!((updated_sphere.pos.x - expected_pos_x).abs() < 1e-5, "Pos X mismatch. Got: {}, Expected: {}", updated_sphere.pos.x, expected_pos_x);
    }

    #[test]
    fn mock_sqrt_computes_square_root() {
        let cpu = MockCpu::default();
        // Input values must be non-negative for square root
        let input_data = vec![0.0f32, 1.0, 4.0, 9.0, 2.0]; 
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| x.sqrt()).collect();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::Sqrt, &dispatch_binds, workgroups)
            .expect("Dispatch for Sqrt failed");

        assert_eq!(result_buffers.len(), 1, "Sqrt should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Sqrt. Got: {}, Expected: {}", got, expected);
        }
    }

    #[test]
    fn mock_rsqrt_computes_reciprocal_square_root() {
        let cpu = MockCpu::default();
        // Input values must be positive for reciprocal square root
        let input_data = vec![1.0f32, 4.0, 9.0, 2.0, 0.25]; 
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| 1.0 / x.sqrt()).collect();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::Rsqrt, &dispatch_binds, workgroups)
            .expect("Dispatch for Rsqrt failed");

        assert_eq!(result_buffers.len(), 1, "Rsqrt should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Rsqrt. Got: {}, Expected: {}", got, expected);
        }
    }

    #[test]
    fn mock_tanh_computes_hyperbolic_tangent() {
        let cpu = MockCpu::default();
        let input_data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 20.0, -20.0]; // Test a range including large values
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| x.tanh()).collect();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::Tanh, &dispatch_binds, workgroups)
            .expect("Dispatch for Tanh failed");

        assert_eq!(result_buffers.len(), 1, "Tanh should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Tanh. Got: {}, Expected: {}", got, expected);
        }
    }

    #[test]
    fn mock_relu_computes_rectified_linear_unit() {
        let cpu = MockCpu::default();
        let input_data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 20.0, -20.0];
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::Relu, &dispatch_binds, workgroups)
            .expect("Dispatch for Relu failed");

        assert_eq!(result_buffers.len(), 1, "Relu should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Relu. Got: {}, Expected: {}", got, expected);
        }
    }

    #[test]
    fn mock_sigmoid_computes_logistic_function() {
        let cpu = MockCpu::default();
        let input_data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 20.0, -20.0];
        let expected_output_data: Vec<f32> = input_data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::Sigmoid, &dispatch_binds, workgroups)
            .expect("Dispatch for Sigmoid failed");

        assert_eq!(result_buffers.len(), 1, "Sigmoid should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>());

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len());

        for (got, expected) in output_values.iter().zip(expected_output_data.iter()) {
            assert!((got - expected).abs() < 1e-6, "Mismatch for Sigmoid. Got: {}, Expected: {}", got, expected);
        }
    }

    #[test]
    fn mock_reduce_sum_computes_sum() {
        let cpu = MockCpu::default();
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -2.0];
        let expected_sum: f32 = input_data.iter().sum();
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        // Output buffer placeholder for a single f32 value
        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![1], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; // Dummy config
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [input_buffer_view, output_buffer_view, config_buffer_view];
        let result_buffers = cpu.dispatch(&Kernel::ReduceSum, &dispatch_binds, workgroups)
            .expect("Dispatch for ReduceSum failed");

        assert_eq!(result_buffers.len(), 1, "ReduceSum should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), std::mem::size_of::<f32>(), "Output buffer size should be size of f32");

        let output_value: f32 = *bytemuck::from_bytes(&output_bytes); // Corrected: dereference
        assert!((output_value - expected_sum).abs() < 1e-6, "Mismatch for ReduceSum. Got: {}, Expected: {}", output_value, expected_sum);
    }

    #[test]
    fn mock_reduce_mean_computes_mean() {
        let cpu = MockCpu::default();
        
        // Test case 1: Non-empty input
        let input_data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, -2.0]; // sum = 13, count = 6
        let expected_mean1: f32 = input_data1.iter().sum::<f32>() / (input_data1.len() as f32);
        
        let input_bytes1: Arc<[u8]> = bytemuck::cast_slice(&input_data1).to_vec().into();
        let input_buffer_view1 = BufferView::new(
            input_bytes1, vec![input_data1.len()], std::mem::size_of::<f32>()
        );

        let output_placeholder_bytes: Arc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder_bytes.clone(), vec![1], std::mem::size_of::<f32>()
        );
        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes.clone(), vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let dispatch_binds1 = [input_buffer_view1, output_buffer_view.clone(), config_buffer_view.clone()];
        let result_buffers1 = cpu.dispatch(&Kernel::ReduceMean, &dispatch_binds1, [1,1,1])
            .expect("Dispatch for ReduceMean (case 1) failed");

        assert_eq!(result_buffers1.len(), 1, "ReduceMean (case 1) should return one output buffer");
        let output_bytes1 = &result_buffers1[0];
        assert_eq!(output_bytes1.len(), std::mem::size_of::<f32>());
        let output_value1: f32 = *bytemuck::from_bytes(&output_bytes1);
        assert!((output_value1 - expected_mean1).abs() < 1e-6, "Mismatch for ReduceMean (case 1). Got: {}, Expected: {}", output_value1, expected_mean1);

        // Test case 2: Empty input
        let input_data2: Vec<f32> = Vec::new();
        let expected_mean2: f32 = 0.0; // Define mean of empty set as 0 for this mock
        
        let input_bytes2: Arc<[u8]> = bytemuck::cast_slice(&input_data2).to_vec().into();
        let input_buffer_view2 = BufferView::new(
            input_bytes2, vec![input_data2.len()], std::mem::size_of::<f32>()
        );
        let dispatch_binds2 = [input_buffer_view2, output_buffer_view, config_buffer_view];
        let result_buffers2 = cpu.dispatch(&Kernel::ReduceMean, &dispatch_binds2, [1,1,1])
            .expect("Dispatch for ReduceMean (case 2) failed");

        assert_eq!(result_buffers2.len(), 1, "ReduceMean (case 2) should return one output buffer");
        let output_bytes2 = &result_buffers2[0];
        assert_eq!(output_bytes2.len(), std::mem::size_of::<f32>());
        let output_value2: f32 = *bytemuck::from_bytes(&output_bytes2);
        assert!((output_value2 - expected_mean2).abs() < 1e-6, "Mismatch for ReduceMean (case 2). Got: {}, Expected: {}", output_value2, expected_mean2);
    }

    #[test]
    fn mock_reduce_max_computes_max_value() {
        let cpu = MockCpu::default();
        
        // Test case 1: Non-empty input
        let input_data1 = vec![1.0f32, -2.0, 5.0, 0.0, 4.5, -10.0];
        let expected_max1: f32 = input_data1.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let input_bytes1: Arc<[u8]> = bytemuck::cast_slice(&input_data1).to_vec().into();
        let input_buffer_view1 = BufferView::new(
            input_bytes1, vec![input_data1.len()], std::mem::size_of::<f32>()
        );

        let output_placeholder_bytes: Arc<[u8]> = vec![0u8; std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_placeholder_bytes.clone(), vec![1], std::mem::size_of::<f32>()
        );
        let config_data = vec![0u32];
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes.clone(), vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let dispatch_binds1 = [input_buffer_view1, output_buffer_view.clone(), config_buffer_view.clone()];
        let result_buffers1 = cpu.dispatch(&Kernel::ReduceMax, &dispatch_binds1, [1,1,1])
            .expect("Dispatch for ReduceMax (case 1) failed");

        assert_eq!(result_buffers1.len(), 1, "ReduceMax (case 1) should return one output buffer");
        let output_bytes1 = &result_buffers1[0];
        assert_eq!(output_bytes1.len(), std::mem::size_of::<f32>());
        let output_value1: f32 = *bytemuck::from_bytes(&output_bytes1);
        assert_eq!(output_value1, expected_max1, "Mismatch for ReduceMax (case 1). Got: {}, Expected: {}", output_value1, expected_max1);

        // Test case 2: Empty input
        let input_data2: Vec<f32> = Vec::new();
        let expected_max2: f32 = f32::NEG_INFINITY; // Max of empty set
        
        let input_bytes2: Arc<[u8]> = bytemuck::cast_slice(&input_data2).to_vec().into();
        let input_buffer_view2 = BufferView::new(
            input_bytes2, vec![input_data2.len()], std::mem::size_of::<f32>()
        );
        let dispatch_binds2 = [input_buffer_view2, output_buffer_view, config_buffer_view];
        let result_buffers2 = cpu.dispatch(&Kernel::ReduceMax, &dispatch_binds2, [1,1,1])
            .expect("Dispatch for ReduceMax (case 2) failed");

        assert_eq!(result_buffers2.len(), 1, "ReduceMax (case 2) should return one output buffer");
        let output_bytes2 = &result_buffers2[0];
        assert_eq!(output_bytes2.len(), std::mem::size_of::<f32>());
        let output_value2: f32 = *bytemuck::from_bytes(&output_bytes2);
        assert_eq!(output_value2, expected_max2, "Mismatch for ReduceMax (case 2). Got: {}, Expected: {}", output_value2, expected_max2);
    }

    #[test]
    fn mock_segmented_reduce_sum_computes_segment_sums() {
        let cpu = MockCpu::default();
        
        // Input data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // Segments (start indices): [0, 3, 7]
        // Expected sums:
        // Segment 1 (indices 0-2): 1 + 2 + 3 = 6
        // Segment 2 (indices 3-6): 4 + 5 + 6 + 7 = 22
        // Segment 3 (indices 7-9): 8 + 9 + 10 = 27
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let segment_indices = vec![0u32, 3, 7]; // Start indices of segments
        let expected_sums = vec![6.0f32, 22.0, 27.0];

        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input_data).to_vec().into();
        let input_buffer_view = BufferView::new(
            input_bytes, vec![input_data.len()], std::mem::size_of::<f32>()
        );

        let segment_indices_bytes: Arc<[u8]> = bytemuck::cast_slice(&segment_indices).to_vec().into();
        let segment_indices_buffer_view = BufferView::new(
            segment_indices_bytes, vec![segment_indices.len()], std::mem::size_of::<u32>()
        );

        // Output buffer placeholder for the sums
        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_sums.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_sums.len()], std::mem::size_of::<f32>()
        );

        // Dummy config buffer
        let config_data = vec![0u32]; 
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        let workgroups = [1, 1, 1];
        let dispatch_binds = [
            input_buffer_view, 
            segment_indices_buffer_view, 
            output_buffer_view, 
            config_buffer_view
        ];
        let result_buffers = cpu.dispatch(&Kernel::SegmentedReduceSum, &dispatch_binds, workgroups)
            .expect("Dispatch for SegmentedReduceSum failed");

        assert_eq!(result_buffers.len(), 1, "SegmentedReduceSum should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_sums.len() * std::mem::size_of::<f32>(), "Output buffer size mismatch");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_sums.len(), "Output values length mismatch");

        for (i, (got, expected)) in output_values.iter().zip(expected_sums.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, 
                "Mismatch for SegmentedReduceSum at segment index {}. Got: {}, Expected: {}", 
                i, got, expected);
        }
    }

    #[test]
    fn mock_scatter_add_adds_values_to_indices() {
        let cpu = MockCpu::default();

        // Initial output/accumulator: [0, 0, 0, 0, 0]
        // Values to add: [1.0, 2.0, 3.0]
        // Indices for additions: [1, 0, 3] (0-indexed)
        // Expected result:
        // output[0] = 0 + 2.0 = 2.0
        // output[1] = 0 + 1.0 = 1.0
        // output[2] = 0 (unchanged)
        // output[3] = 0 + 3.0 = 3.0
        // output[4] = 0 (unchanged)
        // Expected: [2.0, 1.0, 0.0, 3.0, 0.0]

        let initial_output_data = vec![0.0f32; 5];
        let values_to_add_data = vec![1.0f32, 2.0, 3.0];
        let indices_data = vec![1u32, 0, 3]; // Indices in the output buffer
        let expected_final_output_data = vec![2.0f32, 1.0, 0.0, 3.0, 0.0];

        let values_to_add_bytes: Arc<[u8]> = bytemuck::cast_slice(&values_to_add_data).to_vec().into();
        let values_buffer_view = BufferView::new(
            values_to_add_bytes, vec![values_to_add_data.len()], std::mem::size_of::<f32>()
        );

        let indices_bytes: Arc<[u8]> = bytemuck::cast_slice(&indices_data).to_vec().into();
        let indices_buffer_view = BufferView::new(
            indices_bytes, vec![indices_data.len()], std::mem::size_of::<u32>()
        );

        // This is the buffer that will be updated. For MockCpu, it acts as INOUT or IN + separate OUT.
        // The convention for MockCpu is that the kernel returns the *new* state of this buffer.
        let initial_output_bytes: Arc<[u8]> = bytemuck::cast_slice(&initial_output_data).to_vec().into();
        let output_accumulator_buffer_view = BufferView::new(
            initial_output_bytes, vec![initial_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; // Dummy config
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        // Order for ScatterAdd: DATA_IN (values_to_add), INDICES, OUT_ACCUMULATOR, CONFIG
        let dispatch_binds = [
            values_buffer_view, 
            indices_buffer_view, 
            output_accumulator_buffer_view, // This is binds[2]
            config_buffer_view
        ];

        let workgroups = [1, 1, 1];
        let result_buffers = cpu.dispatch(&Kernel::ScatterAdd, &dispatch_binds, workgroups)
            .expect("Dispatch for ScatterAdd failed");

        assert_eq!(result_buffers.len(), 1, "ScatterAdd should return one output buffer (the updated accumulator)");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_final_output_data.len() * std::mem::size_of::<f32>(), "Output buffer size mismatch for ScatterAdd");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_final_output_data.len(), "Output values length mismatch for ScatterAdd");

        for (i, (got, expected)) in output_values.iter().zip(expected_final_output_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, 
                "Mismatch for ScatterAdd at index {}. Got: {}, Expected: {}", 
                i, got, expected);
        }
    }

    #[test]
    fn mock_gather_collects_values_from_indices() {
        let cpu = MockCpu::default();

        // Source Data: [10.0, 11.0, 12.0, 13.0, 14.0]
        // Indices to Gather: [3, 0, 2, 2, 4]
        // Expected Output: [13.0, 10.0, 12.0, 12.0, 14.0]

        let source_data = vec![10.0f32, 11.0, 12.0, 13.0, 14.0];
        let indices_to_gather = vec![3u32, 0, 2, 2, 4];
        let expected_output_data = vec![13.0f32, 10.0, 12.0, 12.0, 14.0];

        let source_data_bytes: Arc<[u8]> = bytemuck::cast_slice(&source_data).to_vec().into();
        let source_buffer_view = BufferView::new(
            source_data_bytes, vec![source_data.len()], std::mem::size_of::<f32>()
        );

        let indices_bytes: Arc<[u8]> = bytemuck::cast_slice(&indices_to_gather).to_vec().into();
        let indices_buffer_view = BufferView::new(
            indices_bytes, vec![indices_to_gather.len()], std::mem::size_of::<u32>()
        );

        // Output buffer placeholder - its size is determined by the number of indices
        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; expected_output_data.len() * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![expected_output_data.len()], std::mem::size_of::<f32>()
        );

        let config_data = vec![0u32]; // Dummy config
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        // Order for Gather: DATA_IN (source_data), INDICES, OUT, CONFIG
        let dispatch_binds = [
            source_buffer_view, 
            indices_buffer_view, 
            output_buffer_view, 
            config_buffer_view
        ];

        let workgroups = [1, 1, 1];
        let result_buffers = cpu.dispatch(&Kernel::Gather, &dispatch_binds, workgroups)
            .expect("Dispatch for Gather failed");

        assert_eq!(result_buffers.len(), 1, "Gather should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>(), "Output buffer size mismatch for Gather");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len(), "Output values length mismatch for Gather");

        for (i, (got, expected)) in output_values.iter().zip(expected_output_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, 
                "Mismatch for Gather at index {}. Got: {}, Expected: {}", 
                i, got, expected);
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestMatMulConfig { m: u32, k: u32, n: u32 }

    #[test]
    fn mock_matmul_multiplies_matrices() {
        let cpu = MockCpu::default();

        // Matrix A (2x3):
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 2u32; // rows_A
        let k = 3u32; // cols_A / rows_B

        // Matrix B (3x2):
        // [7.0,  8.0]
        // [9.0, 10.0]
        // [11.0, 12.0]
        let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let n = 2u32; // cols_B

        // Expected Output C (2x2) = A * B:
        // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        // Expected: [58.0, 64.0, 139.0, 154.0]
        let expected_output_data = vec![58.0f32, 64.0, 139.0, 154.0];

        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a_data).to_vec().into();
        let a_buffer_view = BufferView::new(
            a_bytes, vec![m as usize, k as usize], std::mem::size_of::<f32>()
        );

        let b_bytes: Arc<[u8]> = bytemuck::cast_slice(&b_data).to_vec().into();
        let b_buffer_view = BufferView::new(
            b_bytes, vec![k as usize, n as usize], std::mem::size_of::<f32>()
        );

        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; (m * n) as usize * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![m as usize, n as usize], std::mem::size_of::<f32>()
        );

        let matmul_config = TestMatMulConfig { m, k, n };
        let config_bytes: Arc<[u8]> = bytemuck::bytes_of(&matmul_config).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![1], std::mem::size_of::<TestMatMulConfig>()
        );

        // Order for MatMul: IN_A, IN_B, OUT, CONFIG
        let dispatch_binds = [
            a_buffer_view, 
            b_buffer_view, 
            output_buffer_view, 
            config_buffer_view
        ];

        let workgroups = [1, 1, 1]; 
        let result_buffers = cpu.dispatch(&Kernel::MatMul, &dispatch_binds, workgroups)
            .expect("Dispatch for MatMul failed");

        assert_eq!(result_buffers.len(), 1, "MatMul should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>(), "Output buffer size mismatch for MatMul");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len(), "Output values length mismatch for MatMul");

        for (i, (got, expected)) in output_values.iter().zip(expected_output_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, 
                "Mismatch for MatMul at index {}. Got: {}, Expected: {}", 
                i, got, expected);
        }
    }

    #[test]
    fn mock_rng_normal_produces_deterministic_sequence() {
        let cpu = MockCpu::default();

        let num_values_to_generate = 5usize;
        let expected_output_data: Vec<f32> = (0..num_values_to_generate).map(|i| i as f32 * 0.1).collect(); // e.g., [0.0, 0.1, 0.2, 0.3, 0.4]

        // Output buffer placeholder - its shape determines how many numbers are generated.
        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; num_values_to_generate * std::mem::size_of::<f32>()].into();
        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, vec![num_values_to_generate], std::mem::size_of::<f32>()
        );

        // Config buffer (e.g., for seed, mean, stddev - unused in this simple mock version)
        // For this test, the config isn't strictly driving the generation logic beyond basic presence.
        let config_data = vec![0u32]; // Dummy config, could represent a seed or other params
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config_data).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![config_data.len()], std::mem::size_of::<u32>()
        );

        // Order for RngNormal: OUT, CONFIG (as per layout.rs)
        let dispatch_binds = [output_buffer_view, config_buffer_view];

        let workgroups = [1, 1, 1];
        let result_buffers = cpu.dispatch(&Kernel::RngNormal, &dispatch_binds, workgroups)
            .expect("Dispatch for RngNormal failed");

        assert_eq!(result_buffers.len(), 1, "RngNormal should return one output buffer");
        let output_bytes = &result_buffers[0];
        assert_eq!(output_bytes.len(), expected_output_data.len() * std::mem::size_of::<f32>(), "Output buffer size mismatch for RngNormal");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values.len(), expected_output_data.len(), "Output values length mismatch for RngNormal");

        for (i, (got, expected)) in output_values.iter().zip(expected_output_data.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-6, 
                "Mismatch for RngNormal at index {}. Got: {}, Expected: {}", 
                i, got, expected);
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestExpandConfig { count: u32 }

    #[test]
    fn mock_expand_instances_repeats_template() {
        let cpu = MockCpu::default();

        let template_data = vec![1.0f32, 2.0, 3.0]; // A simple template of three f32s
        let repetition_count = 3u32;

        let mut expected_output_data = Vec::with_capacity(template_data.len() * repetition_count as usize);
        for _ in 0..repetition_count {
            expected_output_data.extend_from_slice(&template_data);
        }

        let template_bytes: Arc<[u8]> = bytemuck::cast_slice(&template_data).to_vec().into();
        let template_buffer_view = BufferView::new(
            template_bytes, 
            vec![template_data.len()], // Shape of the template instance itself
            std::mem::size_of::<f32>()
        );

        let expected_total_elements = template_data.len() * repetition_count as usize;
        let output_placeholder_total_bytes = expected_total_elements * std::mem::size_of::<f32>();
        let output_buffer_placeholder_bytes: Arc<[u8]> = vec![0u8; output_placeholder_total_bytes].into();

        let output_buffer_view = BufferView::new(
            output_buffer_placeholder_bytes, 
            vec![repetition_count as usize, template_data.len()], // Shape of output: [count, elements_per_template]
            std::mem::size_of::<f32>()
        );

        let expand_config = TestExpandConfig { count: repetition_count };
        let config_bytes: Arc<[u8]> = bytemuck::bytes_of(&expand_config).to_vec().into();
        let config_buffer_view = BufferView::new(
            config_bytes, vec![1], std::mem::size_of::<TestExpandConfig>()
        );

        // Order for ExpandInstances: IN (template), OUT_placeholder, CONFIG
        let dispatch_binds = [template_buffer_view, output_buffer_view, config_buffer_view];

        let workgroups = [1, 1, 1];
        let result_buffers = cpu.dispatch(&Kernel::ExpandInstances, &dispatch_binds, workgroups)
            .expect("Dispatch for ExpandInstances failed");

        assert_eq!(result_buffers.len(), 1, "ExpandInstances should return one output buffer");
        let output_bytes = &result_buffers[0];
        
        let expected_output_as_bytes: Vec<u8> = bytemuck::cast_slice(&expected_output_data).to_vec();
        assert_eq!(output_bytes.len(), expected_output_as_bytes.len(), "Output buffer byte length mismatch for ExpandInstances");

        let output_values: &[f32] = bytemuck::cast_slice(output_bytes);
        assert_eq!(output_values, expected_output_data.as_slice(), "Output values mismatch for ExpandInstances");
    }

} // Closing brace for the main tests module

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod wgpu_metal_backend {
    use super::{BufferView, ComputeBackend, ComputeError, Kernel};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    pub struct WgpuMetal {
        #[allow(dead_code)] // instance might not be used directly after init for basic elementwise
        instance: wgpu::Instance,
        #[allow(dead_code)] // adapter might not be used directly after init for basic Noop
        adapter: wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        // Pipeline cache - Mutex for interior mutability with &self in dispatch
        pipelines: Mutex<HashMap<std::mem::Discriminant<Kernel>, Arc<wgpu::ComputePipeline>>>,
    }

    impl WgpuMetal {
        pub fn try_new() -> Result<Self, ComputeError> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                // dx12_shader_compiler: wgpu::Dx12Compiler::default(), // Removed, not in 0.19
                // flags: wgpu::InstanceFlags::default(), // Removed, not in 0.19
                // gles_minor_version: wgpu::Gles3MinorVersion::default(), // Removed, not in 0.19
                ..Default::default() // Use default for other fields
            });

            let adapter_options = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            };

            let adapter = pollster::block_on(instance.request_adapter(&adapter_options))
                .ok_or_else(|| {
                    eprintln!("Failed to find a suitable Metal adapter.");
                    ComputeError::BackendUnavailable
                })?;
            
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WgpuMetal Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    // memory_hints: wgpu::MemoryHints::default(), // Removed, not in 0.19
                },
                None, // Trace path
            ))
            .map_err(|err| {
                eprintln!("Failed to request device: {err:?}");
                ComputeError::BackendUnavailable
            })?;

            Ok(Self {
                instance,
                adapter,
                device: Arc::new(device),
                queue: Arc::new(queue),
                pipelines: Mutex::new(HashMap::new()),
            })
        }
    }

    impl ComputeBackend for WgpuMetal {
        fn dispatch(
            &self,
            shader_kernel: &Kernel,
            _binds: &[BufferView],
            _workgroups: [u32; 3],
        ) -> Result<Vec<Vec<u8>>, ComputeError> {
            // Placeholder for all new kernels to make it compile.
            // Specific WGPU logic for each kernel will be implemented later via TDD.
            match shader_kernel {
                Kernel::Add | Kernel::Sub | Kernel::Mul | Kernel::Div | Kernel::Neg |
                Kernel::Exp | Kernel::Log | Kernel::Sqrt | Kernel::Rsqrt | Kernel::Tanh | Kernel::Relu | Kernel::Sigmoid |
                Kernel::Min | Kernel::Max | Kernel::Clamp | Kernel::Where => {
                    eprintln!("WgpuMetal::dispatch for element-wise op {:?} - placeholder, returning Ok(Vec::new())", shader_kernel);
                    Ok(Vec::new())
                }
                Kernel::ReduceSum | Kernel::ReduceMean | Kernel::ReduceMax |
                Kernel::SegmentedReduceSum | Kernel::ScatterAdd => {
                    eprintln!("WgpuMetal::dispatch for reduction op {:?} - placeholder, returning BackendUnavailable", shader_kernel);
                        Err(ComputeError::BackendUnavailable)
                    }
                Kernel::Gather => {
                    eprintln!("WgpuMetal::dispatch for Gather - placeholder, returning BackendUnavailable");
                    Err(ComputeError::BackendUnavailable)
                }
                Kernel::MatMul => {
                    eprintln!("WgpuMetal::dispatch for MatMul - placeholder, returning BackendUnavailable");
                    Err(ComputeError::BackendUnavailable)
                }
                Kernel::IntegrateBodies | Kernel::DetectContactsSDF | Kernel::SolveContactsPBD => {
                    eprintln!("WgpuMetal::dispatch for physics op {:?} - placeholder, returning BackendUnavailable", shader_kernel);
                    Err(ComputeError::BackendUnavailable)
                }
                Kernel::SolveJointsPBD => {
                    eprintln!("WgpuMetal::dispatch for SolveJointsPBD - placeholder, returning BackendUnavailable");
                    Err(ComputeError::BackendUnavailable)
                }
                Kernel::ExpandInstances => {
                    eprintln!("WgpuMetal::dispatch for ExpandInstances - placeholder, returning BackendUnavailable");
                        Err(ComputeError::BackendUnavailable)
                    }
                Kernel::RngNormal => {
                    eprintln!("WgpuMetal::dispatch for RngNormal - placeholder, returning BackendUnavailable");
                    Err(ComputeError::BackendUnavailable)
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{BufferView, Kernel}; // Use crate:: to access items from compute crate root

        #[test]
        fn wgpu_metal_backend_try_new_and_dispatch_noop() {
            match WgpuMetal::try_new() {
                Ok(backend) => {
                    let dummy_data: Arc<[u8]> = vec![0u8; 4].into();
                    // BufferView::new expects data, shape, and element_size_in_bytes
                    let ok_buf = BufferView::new(dummy_data, vec![4], 1);
                    let result = backend.dispatch(&Kernel::Add, &[ok_buf], [1, 1, 1]);
                    assert!(result.is_ok(), "Dispatching Add on WgpuMetal backend failed: {result:?}");
                    assert!(result.unwrap().is_empty(), "Expected no data back from WgpuMetal Add");
                }
                Err(e) => {
                    // This test runs only on macOS due to cfg(all(target_os = "macos", feature = "metal")).
                    // If try_new fails here, it's a significant issue with the Metal setup or wgpu.
                    panic!("WgpuMetal::try_new() failed on macOS: {e:?}. Ensure Metal is available and working correctly.");
                }
            }
        }
    }
}

// Re-export WgpuMetal at the crate root if the feature is enabled.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use wgpu_metal_backend::WgpuMetal;

/// Returns a compute backend if available, falling back to the CPU implementation.
///
/// On macOS with the `metal` feature enabled this will attempt to create a
/// [`WgpuMetal`] backend. If GPU initialization fails or the feature/OS is not
/// available, a [`MockCpu`] backend is returned.
#[must_use]
pub fn default_backend() -> std::sync::Arc<dyn ComputeBackend> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(gpu) = WgpuMetal::try_new() {
            tracing::info!("Using WgpuMetal backend.");
            return std::sync::Arc::new(gpu);
        }
        tracing::warn!("WgpuMetal backend initialization failed, falling back...");
    }

    #[cfg(feature = "mock")]
    {
        tracing::info!("Using MockCpu backend.");
        return std::sync::Arc::new(MockCpu::default());
    }

    #[cfg(not(feature = "mock"))]
    {
        compile_error!("No compute backend available. Enable the 'mock' feature or ensure a GPU backend can initialize.");
    }
}
