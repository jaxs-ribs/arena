use crate::types::{
    BoxBody, Cylinder, Joint, JointParams, PhysParams, Plane, Sphere, Vec3,
};
use compute::{ComputeBackend, ComputeError};
use std::mem::size_of;
use std::sync::Arc;

/// Final state returned by [`PhysicsSim::run`].
#[derive(Clone, Copy, Debug)]
pub struct SphereState {
    /// Position of the first sphere in the simulation.
    pub pos: Vec3,
}

/// Errors that may occur when running a simulation.
#[derive(Debug)]
pub enum PhysicsError {
    /// Failure in the underlying compute backend.
    BackendError(ComputeError),
    /// Attempted to run a simulation with no spheres present.
    NoSpheres,
}

impl From<ComputeError> for PhysicsError {
    fn from(err: ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

/// Container for all physics objects and simulation parameters.
///
/// [`PhysicsSim`] owns the rigid bodies and provides methods to advance the
/// simulation either on the GPU or CPU. Fields are public for convenience in
/// tests but most users will interact with it via the provided methods.
pub struct PhysicsSim {
    /// Dynamic spheres present in the simulation.
    pub spheres: Vec<Sphere>,
    /// Dynamic axis aligned boxes.
    pub boxes: Vec<BoxBody>,
    /// Dynamic cylinders.
    pub cylinders: Vec<Cylinder>,
    /// Static planes used for collision.
    pub planes: Vec<Plane>,
    /// Global physics parameters.
    pub params: PhysParams,
    /// Distance constraints between spheres.
    pub joints: Vec<Joint>,
    /// Parameters for the joint solver.
    pub joint_params: JointParams,
    backend: Arc<dyn ComputeBackend>,
}

/// Number of threads per workgroup used when dispatching compute shaders.
const WORKGROUP_SIZE: u32 = 256;

impl PhysicsSim {
    /// Creates an empty simulation with default parameters.
    ///
    /// All internal collections are initialised but contain no bodies. The
    /// compute backend defaults to [`compute::default_backend`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            boxes: Vec::new(),
            cylinders: Vec::new(),
            planes: Vec::new(),
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
    /// Convenience constructor for a simulation containing a single sphere.
    ///
    /// The sphere is placed at `initial_height` on the Y axis and has zero
    /// initial velocity. This is primarily used in tests and examples.
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
            boxes: Vec::new(),
            cylinders: Vec::new(),
            planes: Vec::new(),
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

    /// Adds a box to the simulation and returns its index.
    pub fn add_box(&mut self, pos: Vec3, half_extents: Vec3, vel: Vec3) -> usize {
        let index = self.boxes.len();
        self.boxes.push(BoxBody { pos, half_extents, vel });
        index
    }

    /// Adds a cylinder to the simulation and returns its index.
    pub fn add_cylinder(
        &mut self,
        pos: Vec3,
        radius: f32,
        height: f32,
        vel: Vec3,
    ) -> usize {
        let index = self.cylinders.len();
        self.cylinders.push(Cylinder {
            pos,
            vel,
            radius,
            height,
        });
        index
    }

    /// Adds an infinite plane and returns its index.
    pub fn add_plane(&mut self, normal: Vec3, d: f32) -> usize {
        let index = self.planes.len();
        self.planes.push(Plane { normal, d });
        index
    }

    /// Adds a distance joint between two spheres.
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        self.joints.push(Joint {
            body_a,
            body_b,
            rest_length,
            _padding: 0,
        });
    }

    /// Sets an external force for a given sphere.
    ///
    /// The provided force is stored in [`crate::types::PhysParams::forces`] and will be
    /// applied on the next integration step.
    pub fn set_force(&mut self, body_index: usize, force: [f32; 2]) {
        if let Some(f) = self.params.forces.get_mut(body_index) {
            *f = force;
        }
    }

    /// Overrides the compute backend used for dispatching kernels.
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = backend;
    }

    /// Advances the simulation by one step using GPU compute kernels.
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
            _pad: [f32; 3],
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
            _pad: [f32; 3],
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
                    _pad: [0.0; 3],
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

    /// Advances the simulation by one step on the CPU without using the
    /// compute backend.
    pub fn step_cpu(&mut self) {
        let dt = self.params.dt;
        // Spheres
        for (sphere, force) in self.spheres.iter_mut().zip(&self.params.forces) {
            sphere.vel.x += (self.params.gravity.x + force[0]) * dt;
            sphere.vel.y += (self.params.gravity.y + force[1]) * dt;
            sphere.vel.z += self.params.gravity.z * dt;

            sphere.pos.x += sphere.vel.x * dt;
            sphere.pos.y += sphere.vel.y * dt;
            sphere.pos.z += sphere.vel.z * dt;

            for plane in &self.planes {
                let dist = sphere.pos.x * plane.normal.x
                    + sphere.pos.y * plane.normal.y
                    + sphere.pos.z * plane.normal.z
                    + plane.d;
                let radius = 1.0_f32;
                if dist < radius {
                    let correction = radius - dist;
                    sphere.pos.x += plane.normal.x * correction;
                    sphere.pos.y += plane.normal.y * correction;
                    sphere.pos.z += plane.normal.z * correction;
                    let vn = sphere.vel.x * plane.normal.x
                        + sphere.vel.y * plane.normal.y
                        + sphere.vel.z * plane.normal.z;
                    if vn < 0.0 {
                        sphere.vel.x -= vn * plane.normal.x;
                        sphere.vel.y -= vn * plane.normal.y;
                        sphere.vel.z -= vn * plane.normal.z;
                    }
                }
            }
        }

        // Boxes
        for bx in &mut self.boxes {
            bx.vel.x += self.params.gravity.x * dt;
            bx.vel.y += self.params.gravity.y * dt;
            bx.vel.z += self.params.gravity.z * dt;

            bx.pos.x += bx.vel.x * dt;
            bx.pos.y += bx.vel.y * dt;
            bx.pos.z += bx.vel.z * dt;

            for plane in &self.planes {
                let support = bx.half_extents.x.abs() * plane.normal.x.abs()
                    + bx.half_extents.y.abs() * plane.normal.y.abs()
                    + bx.half_extents.z.abs() * plane.normal.z.abs();
                let dist = bx.pos.x * plane.normal.x
                    + bx.pos.y * plane.normal.y
                    + bx.pos.z * plane.normal.z
                    + plane.d
                    - support;
                if dist < 0.0 {
                    let correction = -dist;
                    bx.pos.x += plane.normal.x * correction;
                    bx.pos.y += plane.normal.y * correction;
                    bx.pos.z += plane.normal.z * correction;
                    let vn = bx.vel.x * plane.normal.x
                        + bx.vel.y * plane.normal.y
                        + bx.vel.z * plane.normal.z;
                    if vn < 0.0 {
                        bx.vel.x -= vn * plane.normal.x;
                        bx.vel.y -= vn * plane.normal.y;
                        bx.vel.z -= vn * plane.normal.z;
                    }
                }
            }
        }

        // Cylinders
        for cyl in &mut self.cylinders {
            cyl.vel.x += self.params.gravity.x * dt;
            cyl.vel.y += self.params.gravity.y * dt;
            cyl.vel.z += self.params.gravity.z * dt;

            cyl.pos.x += cyl.vel.x * dt;
            cyl.pos.y += cyl.vel.y * dt;
            cyl.pos.z += cyl.vel.z * dt;

            for plane in &self.planes {
                let support = cyl.radius * (plane.normal.x.abs() + plane.normal.z.abs())
                    + cyl.height * 0.5 * plane.normal.y.abs();
                let dist = cyl.pos.x * plane.normal.x
                    + cyl.pos.y * plane.normal.y
                    + cyl.pos.z * plane.normal.z
                    + plane.d
                    - support;
                if dist < 0.0 {
                    let correction = -dist;
                    cyl.pos.x += plane.normal.x * correction;
                    cyl.pos.y += plane.normal.y * correction;
                    cyl.pos.z += plane.normal.z * correction;
                    let vn = cyl.vel.x * plane.normal.x
                        + cyl.vel.y * plane.normal.y
                        + cyl.vel.z * plane.normal.z;
                    if vn < 0.0 {
                        cyl.vel.x -= vn * plane.normal.x;
                        cyl.vel.y -= vn * plane.normal.y;
                        cyl.vel.z -= vn * plane.normal.z;
                    }
                }
            }
        }
    }

    /// Runs the simulation for `steps` iterations using the GPU backend.
    ///
    /// The time step `dt` is applied before stepping begins. On success the
    /// position of the first sphere is returned for convenience.
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

    /// Runs the simulation for a number of steps on the CPU.
    pub fn run_cpu(&mut self, dt: f32, steps: usize) {
        self.params.dt = dt;
        for _ in 0..steps {
            self.step_cpu();
        }
    }
}
