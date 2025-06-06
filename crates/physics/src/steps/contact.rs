use std::sync::Arc;
use std::mem::size_of;

use compute::ComputeBackend;
use crate::error::PhysicsError;
use crate::types::Sphere;

pub fn detect_and_solve_contacts(
    backend: &dyn ComputeBackend,
    spheres: &mut Vec<Sphere>,
) -> Result<(), PhysicsError> {
    // --- Detect contacts against simple ground plane ---
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SdfVec3 {
        x: f32,
        y: f32,
        z: f32,
        _pad: u32,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SdfBody {
        pos: SdfVec3,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SdfPlane {
        height: f32,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct SdfContact {
        index: u32,
        penetration: f32,
    }

    let bodies: Vec<SdfBody> = spheres
        .iter()
        .map(|s| SdfBody {
            pos: SdfVec3 {
                x: s.pos.x,
                y: s.pos.y,
                z: s.pos.z,
                _pad: 0,
            },
        })
        .collect();

    let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
    let bodies_view =
        compute::BufferView::new(bodies_bytes, vec![bodies.len()], size_of::<SdfBody>());

    let plane = SdfPlane { height: 0.0 };
    let plane_bytes: Arc<[u8]> = bytemuck::bytes_of(&plane).to_vec().into();
    let plane_view = compute::BufferView::new(plane_bytes, vec![1], size_of::<SdfPlane>());

    let placeholder: Arc<[u8]> = vec![0u8; bodies.len() * size_of::<SdfContact>()].into();
    let contacts_view =
        compute::BufferView::new(placeholder, vec![bodies.len()], size_of::<SdfContact>());

    let contact_buffers = backend.dispatch(
        &compute::Kernel::DetectContactsSDF,
        &[bodies_view, plane_view, contacts_view],
        [1, 1, 1],
    )?;

    let contacts: Vec<SdfContact> = if let Some(bytes) = contact_buffers.get(0) {
        bytes
            .chunks_exact(size_of::<SdfContact>())
            .map(bytemuck::pod_read_unaligned)
            .collect()
    } else {
        Vec::new()
    };

    // --- Solve contacts ---
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct PbdContact {
        body_index: u32,
        normal: SdfVec3,
        depth: f32,
    }

    let contacts_pbd: Vec<PbdContact> = contacts
        .iter()
        .map(|c| PbdContact {
            body_index: c.index,
            normal: SdfVec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                _pad: 0,
            },
            depth: c.penetration,
        })
        .collect();

    let contacts_bytes: Arc<[u8]> = bytemuck::cast_slice(&contacts_pbd).to_vec().into();
    let contacts_view = compute::BufferView::new(
        contacts_bytes,
        vec![contacts_pbd.len()],
        size_of::<PbdContact>(),
    );

    let spheres_bytes_arc: Arc<[u8]> = bytemuck::cast_slice(&*spheres).to_vec().into();
    let spheres_view = compute::BufferView::new(
        spheres_bytes_arc,
        vec![spheres.len()],
        size_of::<Sphere>(),
    );

    let params_placeholder: Arc<[u8]> = vec![0u8; 4].into();
    let params_view = compute::BufferView::new(params_placeholder, vec![1], 4);

    let result_buffers = backend.dispatch(
        &compute::Kernel::SolveContactsPBD,
        &[spheres_view, contacts_view, params_view],
        [1, 1, 1],
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