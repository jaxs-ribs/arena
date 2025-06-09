//! # Physics Simulation
//!
//! This module contains the core logic for the JAXS physics engine. It defines
//! the [`PhysicsSim`] struct, which is the main container for all physics
//! objects and simulation parameters. The module provides methods to advance
//! the simulation in time, with support for both CPU and GPU execution.
//!
//! ## Simulation Loop
//!
//! The main simulation loop is driven by the [`run`](PhysicsSim::run) method,
//! which repeatedly calls a stepping function (`step_gpu` or `step_cpu`) to
//! update the state of the rigid bodies. The simulation is designed to be
//! deterministic, meaning that for a given initial state and time step, the
//! simulation will always produce the same result.
//!
//! ## CPU vs. GPU Execution
//!
//! The physics engine can be run in two modes: CPU and GPU. The GPU mode,
//! which is the default, uses the `compute` crate to dispatch compute shaders
//! that perform the physics calculations. This allows for significant
//! performance gains, especially for large numbers of rigid bodies. The CPU
//! mode is primarily used for testing and debugging purposes.

use crate::types::{
    BoxBody, Cylinder, Joint, JointParams, PhysParams, Plane, Sphere, Vec3,
};
use compute::{ComputeBackend, ComputeError};
use std::mem::size_of;
use std::sync::Arc;

/// Represents the final state of a sphere after a simulation run.
///
/// This struct is returned by [`PhysicsSim::run`] and contains the final
/// position of the first sphere in the simulation. It is primarily used for
/// testing and validation purposes.
#[derive(Clone, Copy, Debug)]
pub struct SphereState {
    /// The final position of the first sphere.
    pub pos: Vec3,
}

/// Defines the possible errors that can occur during a physics simulation.
#[derive(Debug)]
pub enum PhysicsError {
    /// An error occurred in the underlying compute backend. This typically
    /// happens during GPU execution and can be caused by issues with the
    /// compute shaders or the GPU device.
    BackendError(ComputeError),
    /// An attempt was made to run a simulation with no spheres. The simulation
    /// requires at least one sphere to be present.
    NoSpheres,
}

impl From<ComputeError> for PhysicsError {
    fn from(err: ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
}

/// The main container for all physics objects and simulation parameters.
///
/// `PhysicsSim` holds all the state required for a physics simulation,
/// including rigid bodies, constraints, and global parameters. It provides a
/// high-level API for setting up and running the simulation.
///
/// The simulation can be advanced step-by-step using the
/// [`step_gpu()`](Self::step_gpu) or [`step_cpu()`](Self::step_cpu) methods,
/// or it can be run for a specified number of steps using the
/// [`run()`](Self::run) or [`run_cpu()`](Self::run_cpu) methods.
///
/// ## Backend
///
/// The `PhysicsSim` uses a `ComputeBackend` to perform the physics
/// calculations. By default, it uses the `wgpu` backend, but this can be
/// customized by calling the [`set_backend()`](Self::set_backend) method.
pub struct PhysicsSim {
    /// The list of dynamic spheres in the simulation.
    pub spheres: Vec<Sphere>,
    /// The list of dynamic, axis-aligned boxes.
    pub boxes: Vec<BoxBody>,
    /// The list of dynamic cylinders.
    pub cylinders: Vec<Cylinder>,
    /// The list of static planes used for collision detection.
    pub planes: Vec<Plane>,
    /// The global parameters for the physics simulation.
    pub params: PhysParams,
    /// The list of distance constraints between spheres.
    pub joints: Vec<Joint>,
    /// The parameters for the joint solver.
    pub joint_params: JointParams,
    backend: Arc<dyn ComputeBackend>,
}

/// Number of threads per workgroup used when dispatching compute shaders.
const WORKGROUP_SIZE: u32 = 256;

impl PhysicsSim {
    /// Creates a new, empty simulation with default parameters.
    ///
    /// This initializes all internal collections for rigid bodies and
    /// constraints, but they will be empty. The compute backend is set to the
    /// default, which is typically `wgpu`.
    ///
    /// # Returns
    ///
    /// A new `PhysicsSim` instance with default settings.
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
    /// Creates a new simulation with a single sphere for testing purposes.
    ///
    /// This is a convenience function that sets up a simple scene with one
    /// sphere at a specified initial height on the Y-axis. The sphere starts
    /// with zero initial velocity. This is primarily used in examples and
    /// integration tests.
    ///
    /// # Arguments
    ///
    /// * `initial_height` - The initial Y-coordinate of the sphere.
    ///
    /// # Returns
    ///
    /// A new `PhysicsSim` instance containing a single sphere.
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

    /// Adds a new sphere to the simulation.
    ///
    /// # Arguments
    ///
    /// * `pos` - The initial position of the sphere.
    /// * `vel` - The initial velocity of the sphere.
    ///
    /// # Returns
    ///
    /// The index of the newly added sphere.
    pub fn add_sphere(&mut self, pos: Vec3, vel: Vec3) -> usize {
        let index = self.spheres.len();
        self.spheres.push(Sphere::new(pos, vel));
        self.params.forces.push([0.0, 0.0]);
        index
    }

    /// Adds a new box to the simulation.
    ///
    /// # Arguments
    ///
    /// * `pos` - The initial position of the box's center.
    /// * `half_extents` - The half-extents of the box, defining its dimensions.
    /// * `vel` - The initial velocity of the box.
    ///
    /// # Returns
    ///
    /// The index of the newly added box.
    pub fn add_box(&mut self, pos: Vec3, half_extents: Vec3, vel: Vec3) -> usize {
        let index = self.boxes.len();
        self.boxes.push(BoxBody { pos, half_extents, vel });
        index
    }

    /// Adds a new cylinder to the simulation.
    ///
    /// # Arguments
    ///
    /// * `pos` - The initial position of the cylinder's center.
    /// * `radius` - The radius of the cylinder's base.
    /// * `height` - The height of the cylinder.
    /// * `vel` - The initial velocity of the cylinder.
    ///
    /// # Returns
    ///
    /// The index of the newly added cylinder.
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

    /// Adds a new static, infinite plane to the simulation.
    ///
    /// # Arguments
    ///
    /// * `normal` - The normal vector of the plane.
    /// * `d` - The distance of the plane from the origin along its normal.
    ///
    /// # Returns
    ///
    /// The index of the newly added plane.
    pub fn add_plane(&mut self, normal: Vec3, d: f32) -> usize {
        let index = self.planes.len();
        self.planes.push(Plane { normal, d });
        index
    }

    /// Adds a new distance joint between two spheres.
    ///
    /// A distance joint constrains two spheres to maintain a fixed distance
    /// from each other.
    ///
    /// # Arguments
    ///
    /// * `body_a` - The index of the first sphere.
    /// * `body_b` - The index of the second sphere.
    /// * `rest_length` - The target distance between the two spheres.
    pub fn add_joint(&mut self, body_a: u32, body_b: u32, rest_length: f32) {
        self.joints.push(Joint {
            body_a,
            body_b,
            rest_length,
            _padding: 0,
        });
    }

    /// Sets an external force to be applied to a specific sphere.
    ///
    /// The provided force will be applied to the sphere on the next
    /// integration step. The force is stored in [`PhysParams::forces`].
    ///
    /// # Arguments
    ///
    /// * `body_index` - The index of the sphere to apply the force to.
    /// * `force` - The force to apply, as a `[f32; 2]` array `[fx, fy]`.
    pub fn set_force(&mut self, body_index: usize, force: [f32; 2]) {
        if let Some(f) = self.params.forces.get_mut(body_index) {
            *f = force;
        }
    }

    /// Overrides the compute backend used for the simulation.
    ///
    /// This allows for switching between different compute backends, such as
    /// `wgpu` or a custom CPU backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - The new compute backend to use.
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = backend;
    }

    /// Advances the simulation by one time step using the GPU.
    ///
    /// This method dispatches a series of compute shaders to perform the
    /// physics calculations on the GPU. The steps are as follows:
    ///
    /// 1.  **Integration:** The positions and velocities of all spheres are
    ///     updated based on the current forces and the time step.
    /// 2.  **Collision Detection:** The GPU checks for collisions between
    ///     spheres and with static planes.
    /// 3.  **Collision Resolution:** The positions and velocities of colliding
    ///     spheres are adjusted to resolve the collisions.
    /// 4.  **Joint Solving:** The constraints imposed by joints are enforced.
    ///
    /// # Returns
    ///
    /// A `Result` indicating whether the step was successful. An error will be
    /// returned if there is an issue with the compute backend.
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

    /// Advances the simulation by one time step using the CPU.
    ///
    /// This method performs the same calculations as
    /// [`step_gpu()`](Self::step_gpu), but it does so on the CPU. It is
    /// primarily used for testing and debugging, as it is significantly slower
    /// than the GPU implementation.
    ///
    /// The CPU step involves the following stages:
    ///
    /// 1.  **Integration:** The position and velocity of each sphere are
    ///     updated based on gravity and external forces.
    /// 2.  **Collision Detection:** The code checks for collisions between each
    ///     pair of spheres and between each sphere and the static planes.
    /// 3.  **Collision Resolution:** Collisions are resolved by adjusting the
    ///     positions and velocities of the colliding spheres.
    /// 4.  **Joint Solving:** The distance constraints for all joints are
    ///     enforced.
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

    /// Runs the simulation for a specified number of steps.
    ///
    /// This method advances the simulation by calling
    /// [`step_gpu()`](Self::step_gpu) repeatedly.
    ///
    /// # Arguments
    ///
    /// * `dt` - The time step for each simulation step.
    /// * `steps` - The number of steps to simulate.
    ///
    /// # Returns
    ///
    /// A `Result` containing the final state of the first sphere, or a
    /// `PhysicsError` if an error occurred during the simulation.
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

    /// Runs the simulation for a specified number of steps on the CPU.
    ///
    /// This method is similar to [`run()`](Self::run), but it uses the CPU for
    /// the simulation. It repeatedly calls [`step_cpu()`](Self::step_cpu) to
    /// advance the simulation.
    ///
    /// # Arguments
    ///
    /// * `dt` - The time step for each simulation step.
    /// * `steps` - The number of steps to simulate.
    pub fn run_cpu(&mut self, dt: f32, steps: usize) {
        if self.spheres.is_empty() {
            return;
        }
        self.params.dt = dt;
        for _ in 0..steps {
            self.step_cpu();
        }
    }
}
