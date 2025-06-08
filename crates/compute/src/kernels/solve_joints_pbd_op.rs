use crate::{BufferView, ComputeError};

pub fn handle_solve_joints_pbd(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "SolveJointsPBD expects at least 3 buffers (bodies, joints, params)",
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
    struct TestBody {
        pos: TestVec3,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestJoint {
        body_a: u32,
        body_b: u32,
        rest_length: f32,
        _padding: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct SolveParams {
        compliance: f32,
        _pad: [f32; 3],
    }

    let bodies_view = &binds[0];
    let joints_view = &binds[1];
    let params_view = &binds[2];

    if params_view.data.len() != std::mem::size_of::<SolveParams>() || params_view.shape != vec![1]
    {
        return Err(ComputeError::ShapeMismatch(
            "Params buffer for SolveJointsPBD has incorrect size or shape",
        ));
    }

    if bodies_view.data.len() % std::mem::size_of::<TestBody>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer size is not a multiple of TestBody size",
        ));
    }
    let num_bodies = bodies_view.data.len() / std::mem::size_of::<TestBody>();
    if bodies_view.shape != vec![num_bodies] {
        return Err(ComputeError::ShapeMismatch(
            "Bodies buffer shape does not match its data length",
        ));
    }

    if joints_view.data.len() % std::mem::size_of::<TestJoint>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Joints buffer size is not a multiple of TestJoint size",
        ));
    }
    let num_joints = joints_view.data.len() / std::mem::size_of::<TestJoint>();
    if joints_view.shape != vec![num_joints] {
        return Err(ComputeError::ShapeMismatch(
            "Joints buffer shape does not match its data length",
        ));
    }

    let mut bodies = bytemuck::cast_slice::<_, TestBody>(&bodies_view.data).to_vec();
    let joints = bytemuck::cast_slice::<_, TestJoint>(&joints_view.data);

    for joint in joints {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        if a >= bodies.len() || b >= bodies.len() {
            return Err(ComputeError::ShapeMismatch("Joint body index out of range"));
        }

        let pa = bodies[a].pos;
        let pb = bodies[b].pos;

        let dx = TestVec3 {
            x: pb.x - pa.x,
            y: pb.y - pa.y,
            z: pb.z - pa.z,
        };
        let len_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        if len_sq == 0.0 {
            continue;
        }
        let len = len_sq.sqrt();
        let diff = (len - joint.rest_length) / len * 0.5;

        bodies[a].pos.x += dx.x * diff;
        bodies[a].pos.y += dx.y * diff;
        bodies[a].pos.z += dx.z * diff;

        bodies[b].pos.x -= dx.x * diff;
        bodies[b].pos.y -= dx.y * diff;
        bodies[b].pos.z -= dx.z * diff;
    }

    let updated_bytes = bytemuck::cast_slice(&bodies).to_vec();
    Ok(vec![updated_bytes])
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use crate::{ComputeBackend, Kernel, backend::mock_cpu::MockCpu};
    use std::sync::Arc as StdArc;

    #[test]
    fn distance_joint_maintains_length() {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestVec3 {
            x: f32,
            y: f32,
            z: f32,
        }

        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestBody {
            pos: TestVec3,
        }

        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestJoint {
            body_a: u32,
            body_b: u32,
            rest_length: f32,
            _padding: u32,
        }

        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct SolveParams {
            compliance: f32,
            _pad: [f32; 3],
        }

        let cpu = MockCpu::default();

        let bodies = vec![
            TestBody {
                pos: TestVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            TestBody {
                pos: TestVec3 {
                    x: 1.5,
                    y: 0.0,
                    z: 0.0,
                },
            },
        ];
        let joints = vec![TestJoint {
            body_a: 0,
            body_b: 1,
            rest_length: 1.0,
            _padding: 0,
        }];
        let params = SolveParams {
            compliance: 0.0,
            _pad: [0.0; 3],
        };

        let body_bytes: StdArc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let joint_bytes: StdArc<[u8]> = bytemuck::cast_slice(&joints).to_vec().into();
        let param_bytes: StdArc<[u8]> = bytemuck::bytes_of(&params).to_vec().into();

        let body_view = BufferView::new(
            body_bytes,
            vec![bodies.len()],
            std::mem::size_of::<TestBody>(),
        );
        let joint_view = BufferView::new(
            joint_bytes,
            vec![joints.len()],
            std::mem::size_of::<TestJoint>(),
        );
        let param_view = BufferView::new(param_bytes, vec![1], std::mem::size_of::<SolveParams>());

        let result = cpu
            .dispatch(
                &Kernel::SolveJointsPBD,
                &[body_view, joint_view, param_view],
                [1, 1, 1],
            )
            .expect("dispatch failed");

        assert_eq!(result.len(), 1);
        let updated: &[TestBody] = bytemuck::cast_slice(&result[0]);
        assert_eq!(updated.len(), 2);

        let dx = updated[1].pos.x - updated[0].pos.x;
        let dy = updated[1].pos.y - updated[0].pos.y;
        let dz = updated[1].pos.z - updated[0].pos.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!((dist - 1.0).abs() < 1e-5, "dist = {dist}");
    }
}
