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
pub struct TestContact {
    pub body_index: u32,
    pub normal: TestVec3,
    pub depth: f32,
    pub _pad: [f32; 3],
}

/// CPU implementation of simple sphere-sphere contact detection.
///
/// Each sphere is assumed to have unit radius. Contacts are emitted in the
/// format expected by the `SolveContactsPBD` kernel.
pub fn handle_detect_contacts_sphere(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 2 {
        return Err(ComputeError::ShapeMismatch(
            "DetectContactsSphere expects 2 buffers (bodies, contacts)",
        ));
    }

    let bodies_view = &binds[0];

    if bodies_view.data.len() % std::mem::size_of::<TestBody>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer size must be a multiple of TestBody",
        ));
    }
    let num_bodies = bodies_view.data.len() / std::mem::size_of::<TestBody>();
    if bodies_view.shape != vec![num_bodies] {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer shape does not match element count",
        ));
    }

    let bodies: &[TestBody] = bytemuck::cast_slice(&bodies_view.data);

    let mut contacts = Vec::<TestContact>::new();
    for i in 0..num_bodies {
        for j in (i + 1)..num_bodies {
            let a = &bodies[i];
            let b = &bodies[j];
            let dx = b.pos.x - a.pos.x;
            let dy = b.pos.y - a.pos.y;
            let dz = b.pos.z - a.pos.z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rad = 1.0f32;
            let rad_sum = rad + rad;
            if dist_sq < rad_sum * rad_sum {
                let dist = dist_sq.sqrt();
                let penetration = rad_sum - dist;
                let normal = if dist > 0.0 {
                    TestVec3 {
                        x: dx / dist,
                        y: dy / dist,
                        z: dz / dist,
                    }
                } else {
                    TestVec3 { x: 1.0, y: 0.0, z: 0.0 }
                };
                // Emit a contact for each body with half penetration depth
                contacts.push(TestContact {
                    body_index: i as u32,
                    normal: TestVec3 {
                        x: -normal.x,
                        y: -normal.y,
                        z: -normal.z,
                    },
                    depth: penetration * 0.5,
                    _pad: [0.0; 3],
                });
                contacts.push(TestContact {
                    body_index: j as u32,
                    normal,
                    depth: penetration * 0.5,
                    _pad: [0.0; 3],
                });
            }
        }
    }

    let out_bytes = bytemuck::cast_slice(&contacts).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use crate::{BufferView, ComputeBackend, CpuBackend, Kernel};
    use super::{TestBody, TestContact, TestVec3};
    use std::sync::Arc;

    #[test]
    fn contacts_generated_for_overlapping_spheres() {
        let cpu = CpuBackend::new();

        let bodies = vec![
            TestBody {
                pos: TestVec3 { x: 0.0, y: 0.0, z: 0.0 },
            },
            TestBody {
                pos: TestVec3 { x: 1.5, y: 0.0, z: 0.0 },
            },
        ];

        let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(bodies_bytes, vec![bodies.len()], std::mem::size_of::<TestBody>());

        let out_placeholder: Arc<[u8]> = vec![0u8; std::mem::size_of::<TestContact>() * 2].into();
        let out_view = BufferView::new(out_placeholder, vec![2], std::mem::size_of::<TestContact>());

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsSphere,
                &[bodies_view, out_view],
                [1, 1, 1],
            )
            .expect("Dispatch failed");

        assert_eq!(result.len(), 1);
        let contacts: &[TestContact] = bytemuck::cast_slice(&result[0]);
        assert_eq!(contacts.len(), 2);
        // contacts should move spheres apart along x axis
        assert_eq!(contacts[0].body_index, 0);
        assert!((contacts[0].normal.x - -1.0).abs() < 1e-6);
        assert!(contacts[0].depth > 0.0);
    }

    #[test]
    fn no_contacts_for_distant_spheres() {
        let cpu = CpuBackend::new();

        let bodies = vec![
            TestBody {
                pos: TestVec3 { x: 0.0, y: 0.0, z: 0.0 },
            },
            TestBody {
                pos: TestVec3 { x: 3.0, y: 0.0, z: 0.0 },
            },
        ];

        let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(bodies_bytes, vec![bodies.len()], std::mem::size_of::<TestBody>());

        let out_placeholder: Arc<[u8]> = vec![0u8; std::mem::size_of::<TestContact>()].into();
        let out_view = BufferView::new(out_placeholder, vec![1], std::mem::size_of::<TestContact>());

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsSphere,
                &[bodies_view, out_view],
                [1, 1, 1],
            )
            .expect("Dispatch failed");

        assert_eq!(result.len(), 1);
        let contacts: &[TestContact] = if result[0].is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&result[0])
        };
        assert!(contacts.is_empty());
    }
}
