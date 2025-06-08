use crate::{BufferView, ComputeError};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestBody {
    pub pos: TestVec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestBox {
    pub center: TestVec3,
    pub half_extents: TestVec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestContact {
    pub body_index: u32,
    pub normal: TestVec3,
    pub depth: f32,
}

/// CPU implementation of sphere-box contact detection for unit spheres.
pub fn handle_detect_contacts_box(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "DetectContactsBox expects 3 buffers (bodies, box, contacts)",
        ));
    }

    let bodies_view = &binds[0];
    let box_view = &binds[1];

    if bodies_view.element_size_in_bytes != core::mem::size_of::<TestBody>() {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer must contain TestBody elements",
        ));
    }

    if box_view.data.len() != core::mem::size_of::<TestBox>() || box_view.shape != vec![1] {
        return Err(ComputeError::ShapeMismatch(
            "Box buffer must contain a single TestBox",
        ));
    }

    if bodies_view.data.len() % core::mem::size_of::<TestBody>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer size must be a multiple of TestBody",
        ));
    }

    let num_bodies = bodies_view.data.len() / core::mem::size_of::<TestBody>();
    if bodies_view.shape != vec![num_bodies] {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer shape does not match element count",
        ));
    }

    let bodies: &[TestBody] = bytemuck::cast_slice(&bodies_view.data);
    let bx: &TestBox = bytemuck::from_bytes(&box_view.data);

    let min_x = bx.center.x - bx.half_extents.x;
    let max_x = bx.center.x + bx.half_extents.x;
    let min_y = bx.center.y - bx.half_extents.y;
    let max_y = bx.center.y + bx.half_extents.y;
    let min_z = bx.center.z - bx.half_extents.z;
    let max_z = bx.center.z + bx.half_extents.z;

    const RAD: f32 = 1.0;

    let mut contacts = Vec::<TestContact>::new();
    for (idx, body) in bodies.iter().enumerate() {
        let px = body.pos.x;
        let py = body.pos.y;
        let pz = body.pos.z;

        let clamped_x = px.clamp(min_x, max_x);
        let clamped_y = py.clamp(min_y, max_y);
        let clamped_z = pz.clamp(min_z, max_z);

        let diff_x = px - clamped_x;
        let diff_y = py - clamped_y;
        let diff_z = pz - clamped_z;

        let dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        if dist_sq > 0.0 {
            if dist_sq < RAD * RAD {
                let dist = dist_sq.sqrt();
                let inv = 1.0 / dist;
                contacts.push(TestContact {
                    body_index: idx as u32,
                    normal: TestVec3 {
                        x: diff_x * inv,
                        y: diff_y * inv,
                        z: diff_z * inv,
                    },
                    depth: RAD - dist,
                });
            }
        } else if px >= min_x
            && px <= max_x
            && py >= min_y
            && py <= max_y
            && pz >= min_z
            && pz <= max_z
        {
            let dist_pos_x = max_x - px;
            let dist_neg_x = px - min_x;
            let dist_pos_y = max_y - py;
            let dist_neg_y = py - min_y;
            let dist_pos_z = max_z - pz;
            let dist_neg_z = pz - min_z;

            let mut min_dist = dist_pos_x;
            let mut normal = TestVec3 { x: 1.0, y: 0.0, z: 0.0 };

            if dist_neg_x < min_dist {
                min_dist = dist_neg_x;
                normal = TestVec3 { x: -1.0, y: 0.0, z: 0.0 };
            }
            if dist_pos_y < min_dist {
                min_dist = dist_pos_y;
                normal = TestVec3 { x: 0.0, y: 1.0, z: 0.0 };
            }
            if dist_neg_y < min_dist {
                min_dist = dist_neg_y;
                normal = TestVec3 { x: 0.0, y: -1.0, z: 0.0 };
            }
            if dist_pos_z < min_dist {
                min_dist = dist_pos_z;
                normal = TestVec3 { x: 0.0, y: 0.0, z: 1.0 };
            }
            if dist_neg_z < min_dist {
                min_dist = dist_neg_z;
                normal = TestVec3 { x: 0.0, y: 0.0, z: -1.0 };
            }

            contacts.push(TestContact {
                body_index: idx as u32,
                normal,
                depth: RAD + min_dist,
            });
        }
    }

    let out_bytes = bytemuck::cast_slice(&contacts).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn contact_generated_for_sphere_touching_box_top() {
        let cpu = CpuBackend::new();

        let bodies = vec![TestBody {
            pos: TestVec3 { x: 0.0, y: 1.5, z: 0.0 },
        }];

        let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(bodies_bytes, vec![bodies.len()], core::mem::size_of::<TestBody>());

        let bx = TestBox {
            center: TestVec3 { x: 0.0, y: 0.0, z: 0.0 },
            half_extents: TestVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let bx_bytes: Arc<[u8]> = bytemuck::bytes_of(&bx).to_vec().into();
        let bx_view = BufferView::new(bx_bytes, vec![1], core::mem::size_of::<TestBox>());

        let out_placeholder: Arc<[u8]> = vec![0u8; core::mem::size_of::<TestContact>()].into();
        let out_view = BufferView::new(out_placeholder, vec![1], core::mem::size_of::<TestContact>());

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsBox,
                &[bodies_view, bx_view, out_view],
                [1, 1, 1],
            )
            .expect("dispatch failed");

        assert_eq!(result.len(), 1);
        let contacts: &[TestContact] = bytemuck::cast_slice(&result[0]);
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].normal.y - 1.0).abs() < 1e-6);
        assert!((contacts[0].depth - 0.5).abs() < 1e-6);
    }

    #[test]
    fn no_contact_for_distant_sphere() {
        let cpu = CpuBackend::new();

        let bodies = vec![TestBody {
            pos: TestVec3 { x: 3.0, y: 0.0, z: 0.0 },
        }];

        let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(bodies_bytes, vec![bodies.len()], core::mem::size_of::<TestBody>());

        let bx = TestBox {
            center: TestVec3 { x: 0.0, y: 0.0, z: 0.0 },
            half_extents: TestVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let bx_bytes: Arc<[u8]> = bytemuck::bytes_of(&bx).to_vec().into();
        let bx_view = BufferView::new(bx_bytes, vec![1], core::mem::size_of::<TestBox>());

        let out_placeholder: Arc<[u8]> = vec![0u8; core::mem::size_of::<TestContact>()].into();
        let out_view = BufferView::new(out_placeholder, vec![1], core::mem::size_of::<TestContact>());

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsBox,
                &[bodies_view, bx_view, out_view],
                [1, 1, 1],
            )
            .expect("dispatch failed");

        assert_eq!(result.len(), 1);
        let contacts: &[TestContact] = if result[0].is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&result[0])
        };
        assert!(contacts.is_empty());
    }
}

