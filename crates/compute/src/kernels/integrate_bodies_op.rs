use crate::{BufferView, ComputeError};

pub fn handle_integrate_bodies(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "IntegrateBodies expects 3 buffers (spheres, params, forces)",
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
    struct TestPhysParams {
        gravity: TestVec3,
        dt: f32,
        _padding1: f32,
        _padding2: f32,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestForce {
        x: f32,
        y: f32,
    }

    let spheres_data_view = &binds[0];
    let params_data_view = &binds[1];
    let forces_data_view = &binds[2];

    if params_data_view.data.len() != std::mem::size_of::<TestPhysParams>()
        || params_data_view.shape != vec![1]
    {
        return Err(ComputeError::ShapeMismatch(
            "Params buffer for IntegrateBodies has incorrect size or shape",
        ));
    }
    let params: &TestPhysParams = bytemuck::from_bytes(&params_data_view.data);

    if spheres_data_view.data.len() % std::mem::size_of::<TestSphere>() != 0 {
        return Err(ComputeError::ShapeMismatch(
            "Spheres buffer size is not a multiple of TestSphere size",
        ));
    }
    let num_spheres = spheres_data_view.data.len() / std::mem::size_of::<TestSphere>();
    if spheres_data_view.shape != vec![num_spheres] {
        return Err(ComputeError::ShapeMismatch(
            "Spheres buffer shape does not match its data length",
        ));
    }

    if forces_data_view.data.len() != num_spheres * std::mem::size_of::<TestForce>()
        || forces_data_view.shape != vec![num_spheres]
    {
        return Err(ComputeError::ShapeMismatch(
            "Forces buffer for IntegrateBodies has incorrect size or shape",
        ));
    }

    let mut updated_spheres =
        bytemuck::cast_slice::<_, TestSphere>(&spheres_data_view.data).to_vec();
    let forces: &[TestForce] = bytemuck::cast_slice(&forces_data_view.data);

    for (sphere, f) in updated_spheres.iter_mut().zip(forces) {
        sphere.vel.x += (params.gravity.x + f.x) * params.dt;
        sphere.vel.y += (params.gravity.y + f.y) * params.dt;
        sphere.vel.z += params.gravity.z * params.dt;

        sphere.pos.x += sphere.vel.x * params.dt;
        sphere.pos.y += sphere.vel.y * params.dt;
        sphere.pos.z += sphere.vel.z * params.dt;

        if sphere.pos.y < 0.0 {
            sphere.pos.y = 0.0;
            sphere.vel.y = 0.0;
        }

        let half_dt = 0.5 * params.dt;
        let ox = sphere.angular_vel.x * half_dt;
        let oy = sphere.angular_vel.y * half_dt;
        let oz = sphere.angular_vel.z * half_dt;
        let qx = sphere.orientation[0];
        let qy = sphere.orientation[1];
        let qz = sphere.orientation[2];
        let qw = sphere.orientation[3];
        sphere.orientation[0] += ox * qw + oy * qz - oz * qy;
        sphere.orientation[1] += oy * qw + oz * qx - ox * qz;
        sphere.orientation[2] += oz * qw + ox * qy - oy * qx;
        sphere.orientation[3] += -ox * qx - oy * qy - oz * qz;
    }

    let updated_spheres_bytes = bytemuck::cast_slice(&updated_spheres).to_vec();
    Ok(vec![updated_spheres_bytes])
}

#[cfg(test)]
mod tests {
    use crate::{BufferView, Kernel, CpuBackend};
    use std::sync::Arc;

    #[test]
    fn mock_integrate_bodies_updates_sphere() {
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
        struct TestPhysParams {
            gravity: TestVec3,
            dt: f32,
            _padding1: f32,
            _padding2: f32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestForce {
            x: f32,
            y: f32,
        }

        let cpu = CpuBackend::new();

        let initial_sphere = TestSphere {
            pos: TestVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            },
            vel: TestVec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: TestVec3 { x: 0.0, y: 0.0, z: 1.0 },
        };
        let spheres_data = vec![initial_sphere];
        let sphere_bytes: StdArc<[u8]> = bytemuck::cast_slice(&spheres_data).to_vec().into();
        let sphere_buffer_view = BufferView::new(
            sphere_bytes,
            vec![spheres_data.len()],
            std::mem::size_of::<TestSphere>(),
        );

        let params = TestPhysParams {
            gravity: TestVec3 {
                x: 0.0,
                y: -9.81,
                z: 0.0,
            },
            dt: 0.1,
            _padding1: 0.0,
            _padding2: 0.0,
        };
        let params_bytes: StdArc<[u8]> = bytemuck::bytes_of(&params).to_vec().into();
        let params_buffer_view =
            BufferView::new(params_bytes, vec![1], std::mem::size_of::<TestPhysParams>());

        let forces = vec![TestForce { x: 0.0, y: 0.0 }];
        let forces_bytes: StdArc<[u8]> = bytemuck::cast_slice(&forces).to_vec().into();
        let forces_buffer_view =
            BufferView::new(forces_bytes, vec![forces.len()], std::mem::size_of::<TestForce>());

        let result_buffers = cpu
            .dispatch(
                &Kernel::IntegrateBodies,
                &[sphere_buffer_view.clone(), params_buffer_view, forces_buffer_view],
                [1, 1, 1],
            )
            .expect("Dispatch for IntegrateBodies failed");

        assert_eq!(result_buffers.len(), 1);
        let updated_spheres_bytes = &result_buffers[0];
        assert_eq!(
            updated_spheres_bytes.len(),
            std::mem::size_of::<TestSphere>() * spheres_data.len()
        );
        let updated_spheres: &[TestSphere] = bytemuck::cast_slice(updated_spheres_bytes);
        assert_eq!(updated_spheres.len(), 1);
        let updated_sphere = updated_spheres[0];

        let expected_vel_y = initial_sphere.vel.y + params.gravity.y * params.dt;
        let expected_pos_y = initial_sphere.pos.y + expected_vel_y * params.dt;
        let expected_vel_x = initial_sphere.vel.x + params.gravity.x * params.dt;
        let expected_pos_x = initial_sphere.pos.x + expected_vel_x * params.dt;

        let expected_orient_z = initial_sphere.orientation[2]
            + initial_sphere.angular_vel.z * params.dt * 0.5 * initial_sphere.orientation[3];

        assert!((updated_sphere.vel.y - expected_vel_y).abs() < 1e-5);
        assert!((updated_sphere.pos.y - expected_pos_y).abs() < 1e-5);
        assert!((updated_sphere.vel.x - expected_vel_x).abs() < 1e-5);
        assert!((updated_sphere.pos.x - expected_pos_x).abs() < 1e-5);
        assert!((updated_sphere.orientation[2] - expected_orient_z).abs() < 1e-5);
    }
}
