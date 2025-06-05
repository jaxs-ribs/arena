#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)] // Allow PhysicsSim in physics crate

// Refined imports
use compute::{ComputeBackend, ComputeError}; // BufferView and Kernel will be fully qualified
use std::mem::size_of; // Specific import for size_of
use std::sync::Arc;

// --- Data Structures ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[must_use] 
pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub pos: Vec3,
    pub vel: Vec3,
}

// This struct will be passed to the shader as uniform.
// WGSL `params: vec4<f32>` where xyz: gravity, w: dt.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhysParams {
    pub gravity: Vec3, // Mapped to params.xyz in shader
    pub dt: f32,       // Mapped to params.w in shader
    pub force: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Joint {
    pub body_a: u32,
    pub body_b: u32,
    pub rest_length: f32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointParams {
    pub compliance: f32,
    pub _pad: [f32; 3],
}

// Structure to return from sim.run() to satisfy the test
pub struct SphereState {
    pub pos: Vec3,
    // pub vel: Vec3, // If needed later
}

// --- Error Type ---

#[derive(Debug)]
pub enum PhysicsError {
    BackendError(ComputeError),
    NoSpheres,
    // Other physics-specific errors can be added here
}

impl From<ComputeError> for PhysicsError {
    fn from(err: ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

// --- Physics Simulator ---

pub struct PhysicsSim {
    pub spheres: Vec<Sphere>, // Host-side copy of sphere data
    pub params: PhysParams,
    pub joints: Vec<Joint>,
    pub joint_params: JointParams,
    backend: Arc<dyn ComputeBackend>, // Using Arc for flexibility, could be Box
                                      // The test calls sim.run() so sim needs to be mutable or run needs &mut self
                                      // If run is &mut self, then backend doesn't strictly need Arc unless sim is cloned
                                      // For now, let's assume backend can be shared or PhysicsSim is not cloned while a run is ongoing.
}

const WORKGROUP_SIZE: u32 = 256;

impl PhysicsSim {
    /// Creates a new simulation with a single sphere at a given initial height.
    #[must_use]
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let sphere = Sphere {
            pos: Vec3::new(0.0, initial_height, 0.0),
            vel: Vec3::new(0.0, 0.0, 0.0),
        };
        let spheres = vec![sphere];

        let params = PhysParams {
            gravity: Vec3::new(0.0, -9.81, 0.0), // Default gravity, will be used by step_gpu
            dt: 0.01, // Default dt, will be overridden by run method's dt
            force: [0.0, 0.0],
        };

        let backend = compute::default_backend();

        Self {
            spheres,
            params,
            joints: Vec::new(),
            joint_params: JointParams { compliance: 0.0, _pad: [0.0; 3] },
            backend,
        }
    }

    pub fn step_gpu(&mut self) -> Result<(), PhysicsError> {
        if self.spheres.is_empty() {
            return Ok(());
        }
        let sphere_bytes_arc: Arc<[u8]> = bytemuck::cast_slice(&self.spheres).to_vec().into();
        let sphere_buffer_view = compute::BufferView::new(
            sphere_bytes_arc.clone(),
            vec![self.spheres.len()],
            size_of::<Sphere>(),
        );

        let params_bytes_arc: Arc<[u8]> = bytemuck::bytes_of(&self.params).to_vec().into();
        let params_buffer_view = compute::BufferView::new(params_bytes_arc, vec![1], size_of::<PhysParams>());

        let num_spheres = self.spheres.len() as u32;
        let workgroups_x = (num_spheres + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let result_buffers = self.backend.dispatch(
            &compute::Kernel::IntegrateBodies,
            &[sphere_buffer_view.clone(), params_buffer_view],
            [workgroups_x, 1, 1],
        )?;

        if let Some(updated) = result_buffers.get(0) {
            if updated.len() == self.spheres.len() * size_of::<Sphere>() {
                let new_spheres: Vec<Sphere> = updated
                    .chunks_exact(size_of::<Sphere>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();
                self.spheres.clone_from_slice(&new_spheres);
            }
        }

        // --- Detect contacts against simple ground plane ---
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct SdfVec3 {
            x: f32,
            y: f32,
            z: f32,
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

        let bodies: Vec<SdfBody> = self
            .spheres
            .iter()
            .map(|s| SdfBody {
                pos: SdfVec3 {
                    x: s.pos.x,
                    y: s.pos.y,
                    z: s.pos.z,
                },
            })
            .collect();

        let bodies_bytes: Arc<[u8]> = bytemuck::cast_slice(&bodies).to_vec().into();
        let bodies_view = compute::BufferView::new(bodies_bytes, vec![bodies.len()], size_of::<SdfBody>());

        let plane = SdfPlane { height: 0.0 };
        let plane_bytes: Arc<[u8]> = bytemuck::bytes_of(&plane).to_vec().into();
        let plane_view = compute::BufferView::new(plane_bytes, vec![1], size_of::<SdfPlane>());

        let placeholder: Arc<[u8]> = vec![0u8; bodies.len() * size_of::<SdfContact>()].into();
        let contacts_view = compute::BufferView::new(placeholder, vec![bodies.len()], size_of::<SdfContact>());

        let contact_buffers = self.backend.dispatch(
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
                normal: SdfVec3 { x: 0.0, y: 1.0, z: 0.0 },
                depth: c.penetration,
            })
            .collect();

        let contacts_bytes: Arc<[u8]> = bytemuck::cast_slice(&contacts_pbd).to_vec().into();
        let contacts_view = compute::BufferView::new(contacts_bytes, vec![contacts_pbd.len()], size_of::<PbdContact>());

        let sphere_bytes_arc: Arc<[u8]> = bytemuck::cast_slice(&self.spheres).to_vec().into();
        let spheres_view = compute::BufferView::new(sphere_bytes_arc.clone(), vec![self.spheres.len()], size_of::<Sphere>());
        let params_placeholder: Arc<[u8]> = vec![0u8; 4].into();
        let params_view = compute::BufferView::new(params_placeholder, vec![1], 4);

        let solved = self.backend.dispatch(
            &compute::Kernel::SolveContactsPBD,
            &[spheres_view, contacts_view, params_view],
            [1, 1, 1],
        )?;

        if let Some(bytes) = solved.get(0) {
            if bytes.len() == self.spheres.len() * size_of::<Sphere>() {
                let updated: Vec<Sphere> = bytes
                    .chunks_exact(size_of::<Sphere>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();
                self.spheres.clone_from_slice(&updated);
            }
        }

        // --- Solve joints ---
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct JointVec3 {
            x: f32,
            y: f32,
            z: f32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct JointBody {
            pos: JointVec3,
        }

        let joint_bodies: Vec<JointBody> = self
            .spheres
            .iter()
            .map(|s| JointBody {
                pos: JointVec3 {
                    x: s.pos.x,
                    y: s.pos.y,
                    z: s.pos.z,
                },
            })
            .collect();

        let body_bytes: Arc<[u8]> = bytemuck::cast_slice(&joint_bodies).to_vec().into();
        let body_view = compute::BufferView::new(body_bytes, vec![joint_bodies.len()], size_of::<JointBody>());

        let joint_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.joints).to_vec().into();
        let joint_view = compute::BufferView::new(joint_bytes, vec![self.joints.len()], size_of::<Joint>());

        let joint_param_bytes: Arc<[u8]> = bytemuck::bytes_of(&self.joint_params).to_vec().into();
        let joint_param_view = compute::BufferView::new(joint_param_bytes, vec![1], size_of::<JointParams>());

        let solved = self.backend.dispatch(
            &compute::Kernel::SolveJointsPBD,
            &[body_view, joint_view, joint_param_view],
            [1, 1, 1],
        )?;

        if let Some(bytes) = solved.get(0) {
            if bytes.len() == self.spheres.len() * size_of::<JointBody>() {
                let updated: Vec<JointBody> = bytes
                    .chunks_exact(size_of::<JointBody>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();
                for (sphere, upd) in self.spheres.iter_mut().zip(updated) {
                    sphere.pos.x = upd.pos.x;
                    sphere.pos.y = upd.pos.y;
                    sphere.pos.z = upd.pos.z;
                }
            }
        }

        Ok(())
    }

    pub fn run(&mut self, dt: f32, steps: usize) -> Result<SphereState, PhysicsError> {
        if self.spheres.is_empty() {
            return Err(PhysicsError::NoSpheres);
        }
        self.params.dt = dt;

        for _ in 0..steps {
            self.step_gpu()?;
        }

        Ok(SphereState { pos: self.spheres[0].pos })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_single_sphere() {
        let sim = PhysicsSim::new_single_sphere(10.0);
        assert_eq!(sim.spheres.len(), 1, "Should have one sphere");
        let sphere = sim.spheres[0];
        assert_eq!(sphere.pos.x, 0.0);
        assert_eq!(sphere.pos.y, 10.0);
        assert_eq!(sphere.pos.z, 0.0);
        assert_eq!(sphere.vel.x, 0.0);
        assert_eq!(sphere.vel.y, 0.0);
        assert_eq!(sphere.vel.z, 0.0);

        // Check params (gravity is default, dt is default but will be set by run())
        assert_eq!(sim.params.gravity.y, -9.81);
        // assert_eq!(sim.params.dt, 0.01); // Initial dt before run()
    }

    // Unit test for step_gpu (will fail or do nothing until step_gpu is implemented)
    #[test]
    fn test_step_gpu_default_backend_ok_with_valid_buffers() {
        let mut sim = PhysicsSim::new_single_sphere(5.0);
        let result = sim.step_gpu();
        assert!(result.is_ok(), "step_gpu should return Ok if buffers are valid, got {result:?}");
        // Further assertions could check if the backend (mock or real GPU) behaved correctly.
        // For now, the mock backend only performs shape checks and returns Ok if shapes are fine.
        // The actual step_gpu implementation will involve creating BufferViews and calling dispatch.
        // This test will become more meaningful once step_gpu does that.
    }

    // Unit test for run with the CPU fallback updating the sphere.
    // #[test]
    // fn test_run_single_sphere_falls_to_ground() {
    //     let mut sim = PhysicsSim::new_single_sphere(10.0);
    //     let dt = 0.01_f32;
    //     let steps = 1000_usize;
    //     let final_state = sim.run(dt, steps).expect("run should succeed");
    //     assert!(final_state.pos.y.abs() < 1e-4, "Sphere should rest on the floor");
    // }
}
