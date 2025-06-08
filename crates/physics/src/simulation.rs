use crate::types::{
    Joint, JointParams, PhysParams, Sphere, Vec3, JOINT_TYPE_DISTANCE,
};
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
            joint_type: JOINT_TYPE_DISTANCE,
            rest_length,
            local_anchor_a: Vec3::new(0.0, 0.0, 0.0),
            local_anchor_b: Vec3::new(0.0, 0.0, 0.0),
            local_axis_a: Vec3::new(0.0, 0.0, 0.0),
            local_axis_b: Vec3::new(0.0, 0.0, 0.0),
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

        // INTEGRATE
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
            _pad1: f32,
            dt: f32,
            _pad2: [f32; 3],
        }
        let raw_params = RawPhysParams {
            gravity: self.params.gravity,
            _pad1: 0.0,
            dt: self.params.dt,
            _pad2: [0.0; 3],
        };
        let params_bytes: Arc<[u8]> = bytemuck::bytes_of(&raw_params).to_vec().into();
        let params_buffer_view =
            compute::BufferView::new(params_bytes, vec![1], size_of::<RawPhysParams>());

        let forces_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.params.forces).to_vec().into();
        let forces_buffer_view = compute::BufferView::new(
            forces_bytes,
            vec![self.params.forces.len()],
            std::mem::size_of::<[f32; 2]>(),
        );

        let num_spheres = self.spheres.len() as u32;
        let workgroups_x = (num_spheres + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let integration_result = self.backend.dispatch(
            &compute::Kernel::IntegrateBodies,
            &[
                sphere_buffer_view,
                params_buffer_view,
                forces_buffer_view,
            ],
            [workgroups_x, 1, 1],
        )?;
        let integrated_spheres_bytes: Arc<[u8]> = if integration_result.is_empty() {
            sphere_bytes
        } else {
            integration_result[0].clone().into()
        };

        // CONTACTS
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ContactVec3 { x: f32, y: f32, z: f32 }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct PbdContact {
            body_index: u32,
            normal: ContactVec3,
            depth: f32,
        }
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct BoxContact {
            body_index: u32,
            normal: ContactVec3,
            depth: f32,
        }

        // Sphere-Sphere
        let sphere_contacts_placeholder: Arc<[u8]> =
            vec![0u8; self.spheres.len() * self.spheres.len() * size_of::<PbdContact>()].into();
        
        let sphere_contact_buffers = self.backend.dispatch(
            &compute::Kernel::DetectContactsSphere,
            &[
                compute::BufferView::new(integrated_spheres_bytes.clone(), vec![self.spheres.len()], size_of::<Sphere>()),
                compute::BufferView::new(sphere_contacts_placeholder.clone(), vec![self.spheres.len() * self.spheres.len()], size_of::<PbdContact>())
            ],
            [1, 1, 1],
        )?;
        let sphere_contacts_bytes: Arc<[u8]> = if sphere_contact_buffers.is_empty() {
            sphere_contacts_placeholder
        } else {
            sphere_contact_buffers[0].clone().into()
        };

        // Sphere-Box
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct BoxShapeInternal {
            center: ContactVec3,
            _pad1: f32,
            half_extents: ContactVec3,
            _pad2: f32,
        }
        let ground_box = BoxShapeInternal {
            center: ContactVec3 { x: 0.0, y: -1.0, z: 0.0 },
            _pad1: 0.0,
            half_extents: ContactVec3 { x: 10.0, y: 1.0, z: 10.0 },
            _pad2: 0.0,
        };
        let box_bytes: Arc<[u8]> = bytemuck::bytes_of(&ground_box).to_vec().into();
        let box_contacts_placeholder: Arc<[u8]> = vec![0u8; self.spheres.len() * size_of::<BoxContact>()].into();

        let box_contact_buffers = self.backend.dispatch(
            &compute::Kernel::DetectContactsBox,
            &[
                compute::BufferView::new(integrated_spheres_bytes.clone(), vec![self.spheres.len()], size_of::<Sphere>()),
                compute::BufferView::new(box_bytes, vec![1], size_of::<BoxShapeInternal>()),
                compute::BufferView::new(box_contacts_placeholder.clone(), vec![self.spheres.len()], size_of::<BoxContact>())
            ],
            [1, 1, 1],
        )?;
        let box_contacts_bytes: Arc<[u8]> = if box_contact_buffers.is_empty() {
            box_contacts_placeholder
        } else {
            box_contact_buffers[0].clone().into()
        };

        // Solve Contacts
        self.backend.dispatch(
            &compute::Kernel::SolveContactsPBD,
            &[
                compute::BufferView::new(integrated_spheres_bytes.clone(), vec![self.spheres.len()], size_of::<Sphere>()),
                compute::BufferView::new(box_contacts_bytes, vec![self.spheres.len()], size_of::<BoxContact>()),
                compute::BufferView::new(sphere_contacts_bytes, vec![self.spheres.len() * self.spheres.len()], size_of::<PbdContact>()),
            ],
            [1, 1, 1],
        )?;

        // JOINTS
        if !self.joints.is_empty() {
            let joint_bytes: Arc<[u8]> = bytemuck::cast_slice(&self.joints).to_vec().into();
            #[repr(C)]
            #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
            struct SolveParams {
                compliance: f32,
                _pad: [f32; 3],
            }
            let solve_params = SolveParams {
                compliance: self.joint_params.compliance,
                _pad: [0.0; 3],
            };
            let params_bytes: Arc<[u8]> = bytemuck::bytes_of(&solve_params).to_vec().into();
    
            self.backend.dispatch(
                &compute::Kernel::SolveJointsPBD,
                &[
                    compute::BufferView::new(integrated_spheres_bytes.clone(), vec![self.spheres.len()], size_of::<Sphere>()),
                    compute::BufferView::new(joint_bytes, vec![self.joints.len()], size_of::<Joint>()),
                    compute::BufferView::new(params_bytes, vec![1], size_of::<SolveParams>())
                ],
                [1, 1, 1],
            )?;
        }

        if integrated_spheres_bytes.len() == self.spheres.len() * size_of::<Sphere>() {
            let new_spheres: Vec<Sphere> = integrated_spheres_bytes
                .chunks_exact(size_of::<Sphere>())
                .map(bytemuck::pod_read_unaligned)
                .collect();
            self.spheres.clone_from_slice(&new_spheres);
        }
        Ok(())
    }

    pub fn run(&mut self, dt: f32, steps: usize) -> Result<SphereState, PhysicsError> {
        if self.spheres.is_empty() {
            return Err(PhysicsError::NoSpheres);
        }
        self.params.dt = dt / steps as f32;

        for _ in 0..steps {
            self.step_gpu()?;
        }

        Ok(SphereState {
            pos: self.spheres[0].pos,
        })
    }
}
