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
    
    // Create buffer views for spheres and parameters
    let sphere_bytes = bytemuck::cast_slice(&sim.spheres);
    let spheres_buffer = BufferView::new(
        Arc::from(sphere_bytes),
        vec![sim.spheres.len()],
        std::mem::size_of::<crate::types::Sphere>(),
    );
    
    // Create a simple params struct that can be Pod
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuParams {
        gravity: [f32; 3],
        dt: f32,
    }
    
    let gpu_params = GpuParams {
        gravity: [sim.params.gravity.x, sim.params.gravity.y, sim.params.gravity.z],
        dt: sim.params.dt,
    };
    
    let params_bytes = bytemuck::bytes_of(&gpu_params);
    let params_buffer = BufferView::new(
        Arc::from(params_bytes),
        vec![1],
        std::mem::size_of::<GpuParams>(),
    );
    
    // Dispatch integration kernel
    let workgroups = calculate_workgroups(sim.spheres.len());
    let results = backend.dispatch(
        &Kernel::IntegrateBodies,
        &[spheres_buffer, params_buffer],
        [workgroups, 1, 1],
    )?;
    
    // Update spheres with results
    if let Some(result_bytes) = results.first() {
        let new_spheres: &[crate::types::Sphere] = bytemuck::cast_slice(result_bytes);
        sim.spheres.copy_from_slice(new_spheres);
    }
    
    Ok(())
}


/// Calculate number of workgroups for GPU dispatch
fn calculate_workgroups(num_elements: usize) -> u32 {
    const WORKGROUP_SIZE: u32 = 256;
    ((num_elements as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE).max(1)
}