use crate::types::{Joint, JointParams, PhysParams, Sphere, Vec3};
use compute::{ComputeBackend, ComputeError};
use std::mem::size_of;
use std::sync::Arc;

pub struct SphereState {
    pub pos: Vec3,
}

#[derive(Debug)]
pub enum PhysicsError {
    BackendError(ComputeError),
    NoSpheres,
}

impl From<ComputeError> for PhysicsError {
    fn from(err: ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

pub struct PhysicsSim {
    pub spheres: Vec<Sphere>,
    pub params: PhysParams,
    pub joints: Vec<Joint>,
    pub joint_params: JointParams,
    backend: Arc<dyn ComputeBackend>,
}

const WORKGROUP_SIZE: u32 = 256;

impl PhysicsSim {
    /// Creates an empty simulation with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            params: PhysParams {
                gravity: Vec3::new(0.0, -9.81, 0.0),
                dt: 0.01,
                forces: Vec::new(),
            },
            joints: Vec::new(),
            joint_params: JointParams {
                compliance: 0.0,
                _pad: [0.0; 3],
            },
            backend: compute::default_backend(),
        }
    }
    #[must_use]
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let sphere = Sphere::new(
            Vec3::new(0.0, initial_height, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        );
        let spheres = vec![sphere];

        let params = PhysParams {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 0.01,
            forces: vec![[0.0, 0.0]],
        };

        let backend = compute::default_backend();

        Self {
            spheres,
            params,
            joints: Vec::new(),
            joint_params: JointParams {
                compliance: 0.0,
                _pad: [0.0; 3],
            },
            backend,
        }
    }

    /// Adds a sphere to the simulation and returns its index.
    pub fn add_sphere(&mut self, pos: Vec3, vel: Vec3) -> usize {
        let index = self.spheres.len();
        self.spheres.push(Sphere::new(pos, vel));
        self.params.forces.push([0.0, 0.0]);
        index
    }

    /// Adds a distance joint between two bodies.
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        self.joints.push(Joint {
            body_a,
            body_b,
            rest_length,
            _padding: 0,
        });
    }

    /// Sets an external force for a given sphere.
    pub fn set_force(&mut self, body_index: usize, force: [f32; 2]) {
        if let Some(f) = self.params.forces.get_mut(body_index) {
            *f = force;
        }
    }

    /// Overrides the compute backend used for dispatching kernels.
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = backend;
    }

    pub fn step_gpu(&mut self) -> Result<(), PhysicsError> {
        if self.spheres.is_empty() {
            return Ok(());
        }
        let sphere_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.spheres).to_vec().into();
        let sphere_buffer_view = compute::BufferView::new(
            sphere_bytes.clone(),
            vec![self.spheres.len()],
            size_of::<Sphere>(),
        );

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct RawPhysParams {
            gravity: Vec3,
            dt: f32,
            _padding1: f32,
            _padding2: f32,
        }
        let raw_params = RawPhysParams {
            gravity: self.params.gravity,
            dt: self.params.dt,
            _padding1: 0.0,
            _padding2: 0.0,
        };
        let params_bytes: Arc<[u8]> = bytemuck::bytes_of(&raw_params).to_vec().into();
        let params_buffer_view = compute::BufferView::new(params_bytes, vec![1], size_of::<RawPhysParams>());

        let forces_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.params.forces).to_vec().into();
        let forces_buffer_view = compute::BufferView::new(
            forces_bytes,
            vec![self.params.forces.len()],
            std::mem::size_of::<[f32; 2]>(),
        );

        let num_spheres = self.spheres.len() as u32;
        let workgroups_x = (num_spheres + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let result_buffers = self.backend.dispatch(
            &compute::Kernel::IntegrateBodies,
            &[sphere_buffer_view.clone(), params_buffer_view, forces_buffer_view],
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

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ContactVec3 {
            x: f32,
            y: f32,
            z: f32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ContactBody {
            pos: ContactVec3,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct PbdContact {
            body_index: u32,
            normal: ContactVec3,
            depth: f32,
        }

        let contact_bodies: Vec<ContactBody> = self
            .spheres
            .iter()
            .map(|s| ContactBody {
                pos: ContactVec3 {
                    x: s.pos.x,
                    y: s.pos.y,
                    z: s.pos.z,
                },
            })
            .collect();

        let body_bytes: Arc<[u8]> = bytemuck::cast_slice(&contact_bodies).to_vec().into();
        let body_view = compute::BufferView::new(body_bytes, vec![contact_bodies.len()], size_of::<ContactBody>());

        let placeholder: Arc<[u8]> =
            vec![0u8; contact_bodies.len() * contact_bodies.len() * size_of::<PbdContact>()].into();
        let contacts_view = compute::BufferView::new(
            placeholder,
            vec![contact_bodies.len() * contact_bodies.len()],
            size_of::<PbdContact>(),
        );

        let sphere_contact_buffers = self.backend.dispatch(
            &compute::Kernel::DetectContactsSphere,
            &[body_view, contacts_view],
            [1, 1, 1],
        )?;

        let mut contacts_pbd: Vec<PbdContact> = if let Some(bytes) = sphere_contact_buffers.get(0) {
            bytes
                .chunks_exact(size_of::<PbdContact>())
                .map(bytemuck::pod_read_unaligned)
                .collect()
        } else {
            Vec::new()
        };

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
        struct BoxVec3 {
            x: f32,
            y: f32,
            z: f32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct BoxShapeInternal {
            center: BoxVec3,
            half_extents: BoxVec3,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct BoxContact {
            body_index: u32,
            normal: BoxVec3,
            depth: f32,
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

        let ground_box = BoxShapeInternal {
            center: BoxVec3 { x: 0.0, y: -1.0, z: 0.0 },
            half_extents: BoxVec3 { x: 5.0, y: 1.0, z: 5.0 },
        };
        let box_bytes: Arc<[u8]> = bytemuck::bytes_of(&ground_box).to_vec().into();
        let box_view = compute::BufferView::new(box_bytes, vec![1], size_of::<BoxShapeInternal>());

        let placeholder: Arc<[u8]> = vec![0u8; bodies.len() * size_of::<BoxContact>()].into();
        let contacts_view = compute::BufferView::new(placeholder, vec![bodies.len()], size_of::<BoxContact>());

        let contact_buffers = self.backend.dispatch(
            &compute::Kernel::DetectContactsBox,
            &[bodies_view, box_view, contacts_view],
            [1, 1, 1],
        )?;

        let contacts_box: Vec<PbdContact> = if let Some(bytes) = contact_buffers.get(0) {
            bytes
                .chunks_exact(size_of::<BoxContact>())
                .map(bytemuck::pod_read_unaligned)
                .map(|c: BoxContact| PbdContact {
                    body_index: c.body_index,
                    normal: ContactVec3 { x: c.normal.x, y: c.normal.y, z: c.normal.z },
                    depth: c.depth,
                })
                .collect()
        } else {
            Vec::new()
        };

        contacts_pbd.extend(contacts_box);

        let contacts_bytes: Arc<[u8]> = bytemuck::cast_slice(&contacts_pbd).to_vec().into();
        let contacts_view = compute::BufferView::new(
            contacts_bytes,
            vec![contacts_pbd.len()],
            size_of::<PbdContact>(),
        );

        let sphere_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.spheres).to_vec().into();
        let spheres_view = compute::BufferView::new(
            sphere_bytes.clone(),
            vec![self.spheres.len()],
            size_of::<Sphere>(),
        );
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

        Ok(SphereState {
            pos: self.spheres[0].pos,
        })
    }
}
