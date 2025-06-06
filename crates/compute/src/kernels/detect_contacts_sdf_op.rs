use crate::{BufferView, ComputeError};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestBody {
    pub pos: TestVec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestPlane {
    pub height: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TestContact {
    pub index: u32,
    pub penetration: f32,
}

/// CPU implementation of contact detection against a very simple SDF.
///
/// The SDF data is interpreted as a single plane parallel to the XZ plane with a
/// `height` field describing its Y position. Bodies penetrating the plane
/// generate contact records containing the body index and the penetration depth.
pub fn handle_detect_contacts_sdf(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "DetectContactsSDF kernel expects 3 buffers (bodies, sdf, contacts)",
        ));
    }

    let bodies_view = &binds[0];
    let sdf_view = &binds[1];

    if bodies_view.element_size_in_bytes != std::mem::size_of::<TestBody>() {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer must contain TestBody elements",
        ));
    }

    if sdf_view.data.len() != std::mem::size_of::<TestPlane>() || sdf_view.shape != vec![1] {
        return Err(ComputeError::ShapeMismatch(
            "SDF buffer must contain a single TestPlane",
        ));
    }

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
    let plane: &TestPlane = bytemuck::from_bytes(&sdf_view.data);

    let mut contacts = Vec::<TestContact>::new();
    for (idx, body) in bodies.iter().enumerate() {
        if body.pos.y < plane.height {
            contacts.push(TestContact {
                index: idx as u32,
                penetration: plane.height - body.pos.y,
            });
        }
    }

    let out_bytes = bytemuck::cast_slice(&contacts).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeBackend, Kernel, MockCpu};
    use std::sync::Arc as StdArc;

    #[test]
    fn contact_generated_for_body_below_plane() {
        let cpu = MockCpu::default();

        let bodies = vec![
            TestBody {
                pos: TestVec3 {
                    x: 0.0,
                    y: -1.0,
                    z: 0.0,
                    _pad: 0,
                },
            },
            TestBody {
                pos: TestVec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    _pad: 0,
                },
            },
        ];

        let bodies_bytes: StdArc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(
            bodies_bytes,
            vec![bodies.len()],
            std::mem::size_of::<TestBody>(),
        );

        let plane = TestPlane { height: 0.0 };
        let plane_bytes: StdArc<[u8]> = bytemuck::bytes_of(&plane).to_vec().into();
        let plane_view = BufferView::new(plane_bytes, vec![1], std::mem::size_of::<TestPlane>());

        let out_placeholder: StdArc<[u8]> = vec![0u8; 16].into();
        let out_view = BufferView::new(
            out_placeholder,
            vec![bodies.len()],
            std::mem::size_of::<TestContact>(),
        );

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsSDF,
                &[bodies_view, plane_view, out_view],
                [1, 1, 1],
            )
            .expect("Dispatch failed");

        assert_eq!(result.len(), 1);
        let contacts: &[TestContact] = bytemuck::cast_slice(&result[0]);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].index, 0);
        assert!((contacts[0].penetration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_contact_for_body_above_plane() {
        let cpu = MockCpu::default();

        let bodies = vec![TestBody {
            pos: TestVec3 {
                x: 0.0,
                y: 0.5,
                z: 0.0,
                _pad: 0,
            },
        }];

        let bodies_bytes: StdArc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = BufferView::new(
            bodies_bytes,
            vec![bodies.len()],
            std::mem::size_of::<TestBody>(),
        );

        let plane = TestPlane { height: 0.0 };
        let plane_bytes: StdArc<[u8]> = bytemuck::bytes_of(&plane).to_vec().into();
        let plane_view = BufferView::new(plane_bytes, vec![1], std::mem::size_of::<TestPlane>());

        let out_placeholder: StdArc<[u8]> = vec![0u8; 8].into();
        let out_view =
            BufferView::new(out_placeholder, vec![1], std::mem::size_of::<TestContact>());

        let result = cpu
            .dispatch(
                &Kernel::DetectContactsSDF,
                &[bodies_view, plane_view, out_view],
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
