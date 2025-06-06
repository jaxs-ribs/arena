use std::sync::Arc;
use std::mem::size_of;

use compute::ComputeBackend;

use crate::error::PhysicsError;
use crate::types::{Sphere, PhysParams};

const WORKGROUP_SIZE: u32 = 256;

pub fn integrate_bodies(
    backend: &dyn ComputeBackend,
    spheres: &mut Vec<Sphere>,
    params: &PhysParams,
) -> Result<(), PhysicsError> {
    if spheres.is_empty() {
        return Ok(());
    }
    let sphere_bytes_arc: Arc<[u8]> = bytemuck::cast_slice(spheres).to_vec().into();
    let sphere_buffer_view = compute::BufferView::new(
        sphere_bytes_arc.clone(),
        vec![spheres.len()],
        size_of::<Sphere>(),
    );

    let params_bytes_arc: Arc<[u8]> = bytemuck::bytes_of(params).to_vec().into();
    let params_buffer_view =
        compute::BufferView::new(params_bytes_arc, vec![1], size_of::<PhysParams>());

    let num_spheres = spheres.len() as u32;
    let workgroups_x = (num_spheres + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let result_buffers = backend.dispatch(
        &compute::Kernel::IntegrateBodies,
        &[sphere_buffer_view.clone(), params_buffer_view],
        [workgroups_x, 1, 1],
    )?;

    if let Some(updated) = result_buffers.get(0) {
        if updated.len() == spheres.len() * size_of::<Sphere>() {
            let new_spheres: Vec<Sphere> = updated
                .chunks_exact(size_of::<Sphere>())
                .map(bytemuck::pod_read_unaligned)
                .collect();
            spheres.clone_from_slice(&new_spheres);
        }
    }
    Ok(())
} 