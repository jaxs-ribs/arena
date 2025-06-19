//! # GPU Execution Pipeline
//! 
//! This module manages the GPU compute pipeline for physics simulation,
//! including buffer management, kernel dispatch, and data synchronization.

use crate::PhysicsSim;
use compute::{BufferView, ComputeError, Kernel};
use std::sync::Arc;

/// Execute one physics step on the GPU
pub fn execute_gpu_step(sim: &mut PhysicsSim) -> Result<(), ComputeError> {
    // For now, just integrate bodies as a simple example
    if !sim.spheres.is_empty() {
        integrate_spheres_gpu(sim)?;
    }
    
    // TODO: Implement other GPU kernels when shaders are ready
    
    Ok(())
}

/// Integrate sphere positions on GPU
fn integrate_spheres_gpu(sim: &mut PhysicsSim) -> Result<(), ComputeError> {
    let backend = &sim.backend;
    
    // Define GPU-compatible sphere struct that matches kernel expectations
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuSphere {
        pos: [f32; 3],
        _pad1: f32,
        vel: [f32; 3],
        _pad2: f32,
        orientation: [f32; 4],
        angular_vel: [f32; 3],
        _pad3: f32,
    }
    
    // Convert our spheres to GPU format
    let gpu_spheres: Vec<GpuSphere> = sim.spheres.iter().map(|s| {
        GpuSphere {
            pos: [s.pos.x, s.pos.y, s.pos.z],
            _pad1: 0.0,
            vel: [s.vel.x, s.vel.y, s.vel.z],
            _pad2: 0.0,
            orientation: s.orientation,
            angular_vel: [s.angular_vel.x, s.angular_vel.y, s.angular_vel.z],
            _pad3: 0.0,
        }
    }).collect();
    
    let sphere_bytes = bytemuck::cast_slice(&gpu_spheres);
    let spheres_buffer = BufferView::new(
        Arc::from(sphere_bytes),
        vec![gpu_spheres.len()],
        std::mem::size_of::<GpuSphere>(),
    );
    
    // Create a simple params struct that can be Pod
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuParams {
        gravity: [f32; 3],
        dt: f32,
        _padding1: f32,
        _padding2: f32,
    }
    
    let gpu_params = GpuParams {
        gravity: [sim.params.gravity.x, sim.params.gravity.y, sim.params.gravity.z],
        dt: sim.params.dt,
        _padding1: 0.0,
        _padding2: 0.0,
    };
    
    let params_bytes = bytemuck::bytes_of(&gpu_params);
    let params_buffer = BufferView::new(
        Arc::from(params_bytes),
        vec![1],
        std::mem::size_of::<GpuParams>(),
    );
    
    // Create forces buffer - ensure we have the right number of forces
    let mut forces = sim.params.forces.clone();
    forces.resize(sim.spheres.len(), [0.0, 0.0]);
    let forces_bytes = bytemuck::cast_slice(&forces);
    let forces_buffer = BufferView::new(
        Arc::from(forces_bytes),
        vec![sim.spheres.len()],
        std::mem::size_of::<[f32; 2]>(),
    );
    
    // Dispatch integration kernel
    let workgroups = calculate_workgroups(sim.spheres.len());
    let results = backend.dispatch(
        &Kernel::IntegrateBodies,
        &[spheres_buffer, params_buffer, forces_buffer],
        [workgroups, 1, 1],
    )?;
    
    // Update spheres with results
    if let Some(result_bytes) = results.first() {
        let new_gpu_spheres: &[GpuSphere] = bytemuck::cast_slice(result_bytes);
        for (sphere, gpu_sphere) in sim.spheres.iter_mut().zip(new_gpu_spheres) {
            sphere.pos.x = gpu_sphere.pos[0];
            sphere.pos.y = gpu_sphere.pos[1];
            sphere.pos.z = gpu_sphere.pos[2];
            sphere.vel.x = gpu_sphere.vel[0];
            sphere.vel.y = gpu_sphere.vel[1];
            sphere.vel.z = gpu_sphere.vel[2];
            sphere.orientation = gpu_sphere.orientation;
            sphere.angular_vel.x = gpu_sphere.angular_vel[0];
            sphere.angular_vel.y = gpu_sphere.angular_vel[1];
            sphere.angular_vel.z = gpu_sphere.angular_vel[2];
        }
    }
    
    Ok(())
}


/// Calculate number of workgroups for GPU dispatch
fn calculate_workgroups(num_elements: usize) -> u32 {
    const WORKGROUP_SIZE: u32 = 256;
    ((num_elements as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1)
}