use crate::{BufferView, ComputeError};

pub fn handle_solve_contacts_pbd(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "SolveContactsPBD expects 3 buffers (bodies, contacts, params)",
        ));
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestVec3 {
        x: f32,
        y: f32,
        z: f32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestSphere {
        pos: TestVec3,
        vel: TestVec3,
        orientation: [f32; 4],
        angular_vel: TestVec3,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestContact {
        body_index: u32,
        normal: TestVec3,
        depth: f32,
    }

    let bodies_view = &binds[0];
    let contacts_view = &binds[1];

    if bodies_view.data.len() % std::mem::size_of::<TestSphere>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer size is not a multiple of TestSphere size",
        ));
    }
    let num_bodies = bodies_view.data.len() / std::mem::size_of::<TestSphere>();
    if bodies_view.shape != vec![num_bodies] {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer shape does not match its data length",
        ));
    }

    if contacts_view.data.len() % std::mem::size_of::<TestContact>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Contacts buffer size is not a multiple of TestContact size",
        ));
    }
    let num_contacts = contacts_view.data.len() / std::mem::size_of::<TestContact>();
    if contacts_view.shape != vec![num_contacts] {
        return Err(ComputeError::ShapeMismatch(
            "Contacts buffer shape does not match its data length",
        ));
    }

    let mut bodies = bytemuck::cast_slice::<_, TestSphere>(&bodies_view.data).to_vec();
    let contacts: &[TestContact] = bytemuck::cast_slice(&contacts_view.data);

    for contact in contacts {
        let idx = contact.body_index as usize;
        if idx >= bodies.len() {
            return Err(ComputeError::ShapeMismatch(
                "Contact body index out of bounds for bodies buffer",
            ));
        }
        let body = &mut bodies[idx];
        body.pos.x += contact.normal.x * contact.depth;
        body.pos.y += contact.normal.y * contact.depth;
        body.pos.z += contact.normal.z * contact.depth;
    }

    let out_bytes = bytemuck::cast_slice(&bodies).to_vec();
    Ok(vec![out_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestVec3 {
        x: f32,
        y: f32,
        z: f32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestSphere {
        pos: TestVec3,
        vel: TestVec3,
        orientation: [f32; 4],
        angular_vel: TestVec3,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestContact {
        body_index: u32,
        normal: TestVec3,
        depth: f32,
    }

    #[test]
    fn mock_solve_contacts_moves_body_out_of_penetration() {
        let cpu = CpuBackend::new();

        let sphere = TestSphere {
            pos: TestVec3 {
                x: 0.0,
                y: -0.1,
                z: 0.0,
            },
            vel: TestVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: TestVec3 { x: 0.0, y: 0.0, z: 0.0 },
        };
        let spheres_bytes: StdArc<[u8]> = bytemuck::bytes_of(&sphere).to_vec().into();
        let spheres_view =
            BufferView::new(spheres_bytes, vec![1], std::mem::size_of::<TestSphere>());

        let contact = TestContact {
            body_index: 0,
            normal: TestVec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            depth: 0.1,
        };
        let contacts_bytes: StdArc<[u8]> = bytemuck::bytes_of(&contact).to_vec().into();
        let contacts_view =
            BufferView::new(contacts_bytes, vec![1], std::mem::size_of::<TestContact>());

        let params_bytes: StdArc<[u8]> = vec![0u8; 4].into();
        let params_view = BufferView::new(params_bytes, vec![1], 4);

        let out = cpu
            .dispatch(
                &Kernel::SolveContactsPBD,
                &[spheres_view, contacts_view, params_view],
                [1, 1, 1],
            )
            .expect("dispatch failed");

        assert_eq!(out.len(), 1);
        let updated_spheres: &[TestSphere] = bytemuck::cast_slice(&out[0]);
        assert_eq!(updated_spheres.len(), 1);
        assert!((updated_spheres[0].pos.y - 0.0).abs() < 1e-6);
    }
}
