use std::sync::Arc;
use std::mem::size_of;

use compute::ComputeBackend;
use crate::error::PhysicsError;
use crate::types::{Sphere, Joint, JointParams, Vec3};

pub fn solve_joints(
    backend: &dyn ComputeBackend,
    spheres: &mut Vec<Sphere>,
    joints: &[Joint],
    joint_params: &JointParams,
) -> Result<(), PhysicsError> {
    if joints.is_empty() {
        return Ok(());
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct JointVec3 {
        x: f32,
        y: f32,
        z: f32,
        _pad: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct JointBody {
        pos: JointVec3,
    }

    let bodies: Vec<JointBody> = spheres
        .iter()
        .map(|s| JointBody {
            pos: JointVec3 {
                x: s.pos.x,
                y: s.pos.y,
                z: s.pos.z,
                _pad: 0,
            },
        })
        .collect();
    
    let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
    let bodies_view = compute::BufferView::new(
        bodies_bytes,
        vec![bodies.len()],
        size_of::<JointBody>(),
    );

    let joints_bytes: Arc<[u8]> = bytemuck::cast_slice(joints).to_vec().into();
    let joints_view = compute::BufferView::new(
        joints_bytes,
        vec![joints.len()],
        size_of::<Joint>(),
    );

    let params_bytes: Arc<[u8]> = bytemuck::bytes_of(joint_params).to_vec().into();
    let params_view = compute::BufferView::new(
        params_bytes,
        vec![1],
        size_of::<JointParams>(),
    );

    let result_buffers = backend.dispatch(
        &compute::Kernel::SolveJointsPBD,
        &[bodies_view, joints_view, params_view],
        [1, 1, 1],
    )?;

    if let Some(updated) = result_buffers.get(0) {
        if updated.len() == spheres.len() * size_of::<JointBody>() {
            let new_bodies: Vec<JointBody> = updated
                .chunks_exact(size_of::<JointBody>())
                .map(bytemuck::pod_read_unaligned)
                .collect();
            for (i, body) in new_bodies.iter().enumerate() {
                spheres[i].pos = Vec3::new(body.pos.x, body.pos.y, body.pos.z);
            }
        }
    }

    Ok(())
} 