// This file will contain the tests for the WGPU backend.
// We will use a "golden master" testing strategy to compare WGPU results against the CPU backend. 

#[cfg(feature = "gpu")]
mod wgpu_tests {
    use compute::{
        CpuBackend, Kernel, BufferView, WgpuBackend, ComputeBackend,
    };
    use std::sync::Arc;

    fn run_kernel_test(kernel: Kernel, inputs: &[BufferView], workgroups: [u32; 3]) {
        let cpu_backend = Arc::new(CpuBackend::new());
        let wgpu_backend = Arc::new(WgpuBackend::new().unwrap());

        let expected = cpu_backend.dispatch(&kernel, inputs, workgroups).unwrap();
        let actual = wgpu_backend.dispatch(&kernel, inputs, workgroups).unwrap();

        assert_eq!(expected.len(), actual.len(), "Mismatched number of output buffers");

        for i in 0..expected.len() {
            assert_eq!(expected[i], actual[i], "Mismatch in buffer at index {}", i);
        }
    }

    #[test]
    fn test_add_kernel() {
        let input1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input2: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let output: Vec<f32> = vec![0.0; 4];

        let input1_bytes: Arc<[u8]> = bytemuck::cast_slice(&input1).to_vec().into();
        let input2_bytes: Arc<[u8]> = bytemuck::cast_slice(&input2).to_vec().into();
        let output_bytes: Arc<[u8]> = bytemuck::cast_slice(&output).to_vec().into();

        let elem_size = std::mem::size_of::<f32>();

        let inputs = vec![
            BufferView::new(input1_bytes, vec![4], elem_size),
            BufferView::new(input2_bytes, vec![4], elem_size),
            BufferView::new(output_bytes, vec![4], elem_size),
        ];

        run_kernel_test(Kernel::Add, &inputs, [1, 1, 1]);
    }

    #[test]
    fn test_matmul_kernel() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix
        let output: Vec<f32> = vec![0.0; 4]; // 2x2 output

        let a_bytes: Arc<[u8]> = bytemuck::cast_slice(&a).to_vec().into();
        let b_bytes: Arc<[u8]> = bytemuck::cast_slice(&b).to_vec().into();
        let output_bytes: Arc<[u8]> = bytemuck::cast_slice(&output).to_vec().into();

        let elem_size = std::mem::size_of::<f32>();

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatMulConfig {
            m: u32,
            k: u32,
            n: u32,
        }
        let config = MatMulConfig { m: 2, k: 3, n: 2 };
        let config_bytes: Arc<[u8]> = bytemuck::bytes_of(&config).to_vec().into();

        let inputs = vec![
            BufferView::new(a_bytes, vec![2, 3], elem_size),
            BufferView::new(b_bytes, vec![3, 2], elem_size),
            BufferView::new(output_bytes, vec![2, 2], elem_size),
            BufferView::new(config_bytes, vec![1], std::mem::size_of::<MatMulConfig>()),
        ];

        run_kernel_test(Kernel::MatMul, &inputs, [1, 1, 1]);
    }

    #[test]
    fn test_reduce_sum_kernel() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output: Vec<f32> = vec![0.0];
        
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let output_bytes: Arc<[u8]> = bytemuck::cast_slice(&output).to_vec().into();

        let elem_size = std::mem::size_of::<f32>();

        let inputs = vec![
            BufferView::new(input_bytes, vec![5], elem_size),
            BufferView::new(output_bytes, vec![1], elem_size),
        ];

        run_kernel_test(Kernel::ReduceSum, &inputs, [1, 1, 1]);
    }

    #[test]
    #[ignore]
    fn test_integrate_bodies_kernel() {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Sphere {
            pos: [f32; 3],
            vel: [f32; 3],
            orientation: [f32; 4],
            angular_vel: [f32; 3],
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PhysParams {
            gravity: [f32; 3],
            dt: f32,
            _padding1: f32,
            _padding2: f32,
        }

        let spheres = vec![
            Sphere { pos: [0.0, 10.0, 0.0], vel: [0.0, 0.0, 0.0], orientation: [0.0, 0.0, 0.0, 1.0], angular_vel: [0.0, 0.0, 0.0] },
            Sphere { pos: [5.0, 5.0, 2.0], vel: [1.0, -1.0, 1.0], orientation: [0.0, 0.0, 0.0, 1.0], angular_vel: [0.0, 0.0, 0.0] },
        ];
        let params = PhysParams {
            gravity: [0.0, -9.81, 0.0],
            dt: 0.01,
            _padding1: 0.0,
            _padding2: 0.0,
        };
        let forces: Vec<[f32; 2]> = vec![[0.0, 0.0]; 2];

        let spheres_bytes: Arc<[u8]> = bytemuck::cast_slice(&spheres).to_vec().into();
        let params_bytes: Arc<[u8]> = bytemuck::bytes_of(&params).to_vec().into();
        let forces_bytes: Arc<[u8]> = bytemuck::cast_slice(&forces).to_vec().into();

        let inputs = vec![
            BufferView::new(spheres_bytes, vec![2], std::mem::size_of::<Sphere>()),
            BufferView::new(params_bytes, vec![1], std::mem::size_of::<PhysParams>()),
            BufferView::new(forces_bytes, vec![2], std::mem::size_of::<[f32; 2]>()),
        ];

        run_kernel_test(Kernel::IntegrateBodies, &inputs, [1, 1, 1]);
    }
    #[test]
    fn test_neg_kernel() {
        let input: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
        let out: Vec<f32> = vec![0.0; 4];
        let bytes_in: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let bytes_out: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(bytes_in, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(bytes_out, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::Neg, &inputs, [1,1,1]);
    }

    #[test]
    fn test_relu_kernel() {
        let input: Vec<f32> = vec![-1.0, 0.5, -0.2, 3.0];
        let out: Vec<f32> = vec![0.0; 4];
        let bytes_in: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let bytes_out: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(bytes_in, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(bytes_out, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::Relu, &inputs, [1,1,1]);
    }

    #[test]
    fn test_reduce_mean_kernel() {
        let input: Vec<f32> = vec![1.0,2.0,3.0,4.0];
        let out: Vec<f32> = vec![0.0];
        let bytes_in: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let bytes_out: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(bytes_in, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(bytes_out, vec![1], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::ReduceMean, &inputs, [1,1,1]);
    }

    #[test]
    fn test_gather_kernel() {
        let data: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
        let idx: Vec<u32> = vec![2,0,3];
        let out: Vec<f32> = vec![0.0; 3];
        let data_b: Arc<[u8]> = bytemuck::cast_slice(&data).to_vec().into();
        let idx_b: Arc<[u8]> = bytemuck::cast_slice(&idx).to_vec().into();
        let out_b: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(data_b, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(idx_b, vec![3], std::mem::size_of::<u32>()),
            BufferView::new(out_b, vec![3], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::Gather, &inputs, [1,1,1]);
    }

    #[test]
    fn test_scatter_add_kernel() {
        let values: Vec<f32> = vec![1.0,2.0,3.0];
        let indices: Vec<u32> = vec![1,0,2];
        let acc: Vec<f32> = vec![0.0,0.0,0.0];
        let values_b: Arc<[u8]> = bytemuck::cast_slice(&values).to_vec().into();
        let idx_b: Arc<[u8]> = bytemuck::cast_slice(&indices).to_vec().into();
        let acc_b: Arc<[u8]> = bytemuck::cast_slice(&acc).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(values_b, vec![3], std::mem::size_of::<f32>()),
            BufferView::new(idx_b, vec![3], std::mem::size_of::<u32>()),
            BufferView::new(acc_b, vec![3], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::ScatterAdd, &inputs, [1,1,1]);
    }

    #[test]
    fn test_expand_instances_kernel() {
        let template: Vec<f32> = vec![1.0,2.0];
        let out: Vec<f32> = vec![0.0; 4];
        let template_b: Arc<[u8]> = bytemuck::cast_slice(&template).to_vec().into();
        let out_b: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg_val: u32 = 2;
        let cfg_b: Arc<[u8]> = bytemuck::bytes_of(&cfg_val).to_vec().into();
        let inputs = vec![
            BufferView::new(template_b, vec![2], std::mem::size_of::<f32>()),
            BufferView::new(out_b, vec![2,2], std::mem::size_of::<f32>()),
            BufferView::new(cfg_b, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::ExpandInstances, &inputs, [1,1,1]);
    }

    #[test]
    fn test_rng_normal_kernel() {
        let out: Vec<f32> = vec![0.0; 4];
        let out_b: Arc<[u8]> = bytemuck::cast_slice(&out).to_vec().into();
        let cfg: Arc<[u8]> = bytemuck::cast_slice(&[0u32]).to_vec().into();
        let inputs = vec![
            BufferView::new(out_b, vec![4], std::mem::size_of::<f32>()),
            BufferView::new(cfg, vec![1], std::mem::size_of::<u32>()),
        ];
        run_kernel_test(Kernel::RngNormal, &inputs, [1,1,1]);
    }
}
