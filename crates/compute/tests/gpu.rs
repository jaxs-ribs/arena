// This file will contain the tests for the WGPU backend.
// We will use a "golden master" testing strategy to compare WGPU results against the CPU backend. 

#[cfg(feature = "gpu")]
mod wgpu_tests {
    use compute::{
        default_backend, CpuBackend, Kernel, BufferView, WgpuBackend,
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
        let output: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let input1_bytes: Arc<[u8]> = bytemuck::cast_slice(&input1).to_vec().into();
        let input2_bytes: Arc<[u8]> = bytemuck::cast_slice(&input2).to_vec().into();
        let output_bytes: Arc<[u8]> = bytemuck::cast_slice(&output).to_vec().into();

        let elem_size = std::mem::size_of::<f32>();

        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&[0.0f32; 0]).to_vec().into();
        let inputs = vec![
            BufferView::new(input1_bytes, vec![4], elem_size),
            BufferView::new(input2_bytes, vec![4], elem_size),
            BufferView::new(output_bytes, vec![4], elem_size),
            BufferView::new(config_bytes, vec![0], elem_size),
        ];

        run_kernel_test(Kernel::Add, &inputs, [1, 1, 1]);
    }

    #[test]
    fn test_matmul_kernel() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let input: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3 matrix (batch of 2, 3-element vectors)
        let output: Vec<f32> = vec![0.0; 4]; // 2x2 output

        let weights_bytes: Arc<[u8]> = bytemuck::cast_slice(&weights).to_vec().into();
        let input_bytes: Arc<[u8]> = bytemuck::cast_slice(&input).to_vec().into();
        let output_bytes: Arc<[u8]> = bytemuck::cast_slice(&output).to_vec().into();

        let elem_size = std::mem::size_of::<f32>();
        
        let config: Vec<u32> = vec![2, 3, 2]; // M, N, K
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&config).to_vec().into();

        let inputs = vec![
            BufferView::new(weights_bytes, vec![2, 3], elem_size),
            BufferView::new(input_bytes, vec![2, 3], elem_size),
            BufferView::new(output_bytes, vec![2, 2], elem_size),
            BufferView::new(config_bytes, vec![3], std::mem::size_of::<u32>()),
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
        let config_bytes: Arc<[u8]> = bytemuck::cast_slice(&[0.0f32; 0]).to_vec().into();

        let inputs = vec![
            BufferView::new(input_bytes, vec![5], elem_size),
            BufferView::new(output_bytes, vec![1], elem_size),
            BufferView::new(config_bytes, vec![0], elem_size),
        ];

        run_kernel_test(Kernel::ReduceSum, &inputs, [1, 1, 1]);
    }

    #[test]
    fn test_integrate_bodies_kernel() {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Sphere {
            pos: [f32; 3],
            vel: [f32; 3],
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PhysParams {
            gravity: [f32; 3],
            dt: f32,
        }

        let spheres = vec![
            Sphere { pos: [0.0, 10.0, 0.0], vel: [0.0, 0.0, 0.0] },
            Sphere { pos: [5.0, 5.0, 2.0], vel: [1.0, -1.0, 1.0] },
        ];
        let params = PhysParams { gravity: [0.0, -9.81, 0.0], dt: 0.01 };

        let spheres_bytes: Arc<[u8]> = bytemuck::cast_slice(&spheres).to_vec().into();
        let params_bytes: Arc<[u8]> = bytemuck::cast_slice(&[params]).to_vec().into();

        let inputs = vec![
            BufferView::new(spheres_bytes, vec![2], std::mem::size_of::<Sphere>()),
            BufferView::new(params_bytes, vec![1], std::mem::size_of::<PhysParams>()),
        ];
        
        run_kernel_test(Kernel::IntegrateBodies, &inputs, [1, 1, 1]);
    }
} 