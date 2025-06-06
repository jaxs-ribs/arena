use crate::{BufferView, ComputeError};

pub fn handle_integrate_bodies(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 2 {
        return Err(ComputeError::ShapeMismatch(
            "IntegrateBodies expects at least 2 buffers (spheres, params)",
        ));
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestVec3 {
        x: f32,
        y: f32,
        z: f32,
        _pad: u32,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestSphere {
        pos: TestVec3,
        vel: TestVec3,
        radius: f32,
        _pad: [u32; 3],
    }
    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestPhysParams {
        gravity: TestVec3,
        dt: f32,
        _pad: [u32; 3],
    }

    let spheres_data_view = &binds[0];
    let params_data_view = &binds[1];

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

    let mut updated_spheres =
        bytemuck::cast_slice::<_, TestSphere>(&spheres_data_view.data).to_vec();

    for sphere in &mut updated_spheres {
        sphere.vel.x += params.gravity.x * params.dt;
        sphere.vel.y += params.gravity.y * params.dt;
        sphere.vel.z += params.gravity.z * params.dt;

        sphere.pos.x += sphere.vel.x * params.dt;
        sphere.pos.y += sphere.vel.y * params.dt;
        sphere.pos.z += sphere.vel.z * params.dt;

        if sphere.pos.y < 0.0 {
            sphere.pos.y = 0.0;
            sphere.vel.y = 0.0;
        }
    }

    let updated_spheres_bytes = bytemuck::cast_slice(&updated_spheres).to_vec();
    Ok(vec![updated_spheres_bytes])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeBackend, Kernel, MockCpu};
    use std::sync::Arc as StdArc;

    #[test]
    fn mock_integrate_bodies_updates_sphere() {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestVec3 {
            x: f32,
            y: f32,
            z: f32,
            _pad: u32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestSphere {
            pos: TestVec3,
            vel: TestVec3,
            radius: f32,
            _pad: [u32; 3],
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        struct TestPhysParams {
            gravity: TestVec3,
            dt: f32,
            _pad: [u32; 3],
        }

        let cpu = MockCpu::default();

        let initial_sphere = TestSphere {
            pos: TestVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
                _pad: 0,
            },
            vel: TestVec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
                _pad: 0,
            },
            radius: 1.0,
            _pad: [0; 3],
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
                _pad: 0,
            },
            dt: 0.1,
            _pad: [0; 3],
        };
        let params_bytes: StdArc<[u8]> = bytemuck::bytes_of(&params).to_vec().into();
        let params_buffer_view =
            BufferView::new(params_bytes, vec![1], std::mem::size_of::<TestPhysParams>());

        let result_buffers = cpu
            .dispatch(
                &Kernel::IntegrateBodies,
                &[sphere_buffer_view.clone(), params_buffer_view],
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

        assert!((updated_sphere.vel.y - expected_vel_y).abs() < 1e-5);
        assert!((updated_sphere.pos.y - expected_pos_y).abs() < 1e-5);
        assert!((updated_sphere.vel.x - expected_vel_x).abs() < 1e-5);
        assert!((updated_sphere.pos.x - expected_pos_x).abs() < 1e-5);
    }
}
