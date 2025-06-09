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
    BoundingBox, BoxBody, ContactDebugInfo, Cylinder, ForceDebugInfo, Joint, JointParams, 
    Material, PhysicsDebugInfo, PhysParams, Plane, Sphere, SpatialGrid, SpatialGridDebugInfo, 
    Vec3, VelocityDebugInfo,
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
    /// Spatial grid for broad-phase collision detection.
    pub spatial_grid: SpatialGrid,
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
        // Create a reasonable default spatial grid
        // Grid covers a 100x100x100 world with 4-unit cells (good for ~1 unit radius spheres)
        let bounds = BoundingBox {
            min: Vec3::new(-50.0, -10.0, -50.0),
            max: Vec3::new(50.0, 90.0, 50.0),
        };
        let spatial_grid = SpatialGrid::new(4.0, bounds);
        
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
            spatial_grid,
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
            1.0, // Default radius of 1.0
        );
        let spheres = vec![sphere];

        let params = PhysParams {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 0.01,
            forces: vec![[0.0, 0.0]],
        };

        let bounds = BoundingBox {
            min: Vec3::new(-50.0, -10.0, -50.0),
            max: Vec3::new(50.0, 90.0, 50.0),
        };
        let spatial_grid = SpatialGrid::new(4.0, bounds);

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
            spatial_grid,
            backend,
        }
    }

    /// Adds a new sphere to the simulation.
    ///
    /// # Arguments
    ///
    /// * `pos` - The initial position of the sphere.
    /// * `vel` - The initial velocity of the sphere.
    /// * `radius` - The radius of the sphere.
    ///
    /// # Returns
    ///
    /// The index of the newly added sphere.
    pub fn add_sphere(&mut self, pos: Vec3, vel: Vec3, radius: f32) -> usize {
        let index = self.spheres.len();
        self.spheres.push(Sphere::new(pos, vel, radius));
        self.params.forces.push([0.0, 0.0]);
        index
    }

    /// Adds a new sphere with custom material to the simulation.
    pub fn add_sphere_with_material(&mut self, pos: Vec3, vel: Vec3, radius: f32, material: Material) -> usize {
        let index = self.spheres.len();
        self.spheres.push(Sphere::with_material(pos, vel, radius, material));
        self.params.forces.push([0.0, 0.0]);
        index
    }

    /// Adds a new sphere with custom mass and material to the simulation.
    pub fn add_sphere_with_mass_and_material(&mut self, pos: Vec3, vel: Vec3, radius: f32, mass: f32, material: Material) -> usize {
        let index = self.spheres.len();
        self.spheres.push(Sphere::with_mass_and_material(pos, vel, radius, mass, material));
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
        self.boxes.push(BoxBody { pos, half_extents, vel, material: Material::default() });
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
            material: Material::default(),
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

    /// Configures the spatial grid used for broad-phase collision detection.
    ///
    /// # Arguments
    ///
    /// * `cell_size` - Size of each grid cell (should be roughly 2x the average object radius)
    /// * `bounds` - World-space bounds that the grid covers
    pub fn configure_spatial_grid(&mut self, cell_size: f32, bounds: BoundingBox) {
        self.spatial_grid = SpatialGrid::new(cell_size, bounds);
    }

    /// Returns statistics about the spatial grid performance.
    pub fn spatial_grid_stats(&self) -> (usize, usize, f32) {
        let total_cells = self.spatial_grid.cells.len();
        let occupied_cells = self.spatial_grid.cells.iter().filter(|cell| !cell.is_empty()).count();
        let occupancy_ratio = if total_cells > 0 { occupied_cells as f32 / total_cells as f32 } else { 0.0 };
        (total_cells, occupied_cells, occupancy_ratio)
    }

    /// Generates debug information for visualization.
    pub fn get_debug_info(&self) -> PhysicsDebugInfo {
        let mut debug_info = PhysicsDebugInfo {
            contacts: Vec::new(),
            velocity_vectors: Vec::new(),
            force_vectors: Vec::new(),
            spatial_grid_info: SpatialGridDebugInfo {
                cell_size: self.spatial_grid.cell_size,
                bounds: self.spatial_grid.bounds,
                dimensions: self.spatial_grid.dimensions,
                occupied_cells: Vec::new(),
            },
        };

        // Collect velocity vectors for all spheres
        for (i, sphere) in self.spheres.iter().enumerate() {
            let speed = (sphere.vel.x * sphere.vel.x + sphere.vel.y * sphere.vel.y + sphere.vel.z * sphere.vel.z).sqrt();
            if speed > 0.01 { // Only show significant velocities
                debug_info.velocity_vectors.push(VelocityDebugInfo {
                    position: sphere.pos,
                    velocity: sphere.vel,
                    object_index: i,
                });
            }
        }

        // Collect force vectors (external forces applied to spheres)
        for (i, force) in self.params.forces.iter().enumerate() {
            if i < self.spheres.len() {
                let force_magnitude = (force[0] * force[0] + force[1] * force[1]).sqrt();
                if force_magnitude > 0.01 { // Only show significant forces
                    debug_info.force_vectors.push(ForceDebugInfo {
                        position: self.spheres[i].pos,
                        force: Vec3::new(force[0], force[1], 0.0),
                        object_index: i,
                    });
                }
            }
        }

        // Collect spatial grid occupation data
        for (cell_index, cell) in self.spatial_grid.cells.iter().enumerate() {
            if !cell.is_empty() {
                debug_info.spatial_grid_info.occupied_cells.push((cell_index, cell.len()));
            }
        }

        // TODO: Collect contact information during collision detection
        // This would require modifying the collision detection to store contact data

        debug_info
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
            // F = ma, so a = F/m. For gravity, F = mg, so a = g (mass cancels out)
            // External forces are assumed to be actual forces, so need to divide by mass
            sphere.vel.x += (self.params.gravity.x + force[0] / sphere.mass) * dt;
            sphere.vel.y += (self.params.gravity.y + force[1] / sphere.mass) * dt;
            sphere.vel.z += self.params.gravity.z * dt;

            sphere.pos.x += sphere.vel.x * dt;
            sphere.pos.y += sphere.vel.y * dt;
            sphere.pos.z += sphere.vel.z * dt;

            for plane in &self.planes {
                let dist = sphere.pos.x * plane.normal.x
                    + sphere.pos.y * plane.normal.y
                    + sphere.pos.z * plane.normal.z
                    + plane.d;
                let radius = sphere.radius;
                let contact_threshold = radius + 0.01; // Small tolerance for resting contact
                
                if dist < contact_threshold {
                    let is_penetrating = dist < radius;
                    
                    if is_penetrating {
                        // Position correction for penetration
                        let correction = radius - dist;
                        sphere.pos.x += plane.normal.x * correction;
                        sphere.pos.y += plane.normal.y * correction;
                        sphere.pos.z += plane.normal.z * correction;
                    }
                    
                    // Velocity-based collision and friction response
                    let vn = sphere.vel.x * plane.normal.x
                        + sphere.vel.y * plane.normal.y
                        + sphere.vel.z * plane.normal.z;
                    
                    // Apply restitution only for actual collisions (negative velocity into surface)
                    if vn < 0.0 && is_penetrating {
                        let restitution = sphere.material.restitution;
                        let new_vn = -vn * restitution;
                        let velocity_change = new_vn - vn;
                        
                        sphere.vel.x += velocity_change * plane.normal.x;
                        sphere.vel.y += velocity_change * plane.normal.y;
                        sphere.vel.z += velocity_change * plane.normal.z;
                    }
                    
                    // Apply friction for both colliding and resting objects
                    let friction = sphere.material.friction;
                    
                    // Calculate current velocity normal component
                    let vel_normal = sphere.vel.x * plane.normal.x + 
                                   sphere.vel.y * plane.normal.y + 
                                   sphere.vel.z * plane.normal.z;
                    
                    // Get tangential velocity (velocity minus normal component)
                    let tangent_vel_x = sphere.vel.x - vel_normal * plane.normal.x;
                    let tangent_vel_y = sphere.vel.y - vel_normal * plane.normal.y;
                    let tangent_vel_z = sphere.vel.z - vel_normal * plane.normal.z;
                    
                    let tangent_speed = (tangent_vel_x * tangent_vel_x + 
                                       tangent_vel_y * tangent_vel_y + 
                                       tangent_vel_z * tangent_vel_z).sqrt();
                    
                    if tangent_speed > 1e-6 {
                        // Calculate normal force magnitude (for friction calculation)
                        let normal_force = sphere.mass * (-self.params.gravity.x * plane.normal.x 
                                                        - self.params.gravity.y * plane.normal.y
                                                        - self.params.gravity.z * plane.normal.z).max(0.0);
                        
                        // Maximum friction force (Coulomb friction law: f_max = μ * N)
                        let max_friction_force = friction * normal_force;
                        
                        // Required force to stop tangential motion this timestep
                        let required_force = tangent_speed * sphere.mass / dt;
                        
                        // Apply the minimum of required force and maximum friction
                        let applied_friction = max_friction_force.min(required_force);
                        let friction_factor = if required_force > 1e-6 {
                            1.0 - (applied_friction / required_force)
                        } else {
                            0.0
                        };
                        
                        // Apply friction damping
                        sphere.vel.x = vel_normal * plane.normal.x + tangent_vel_x * friction_factor;
                        sphere.vel.y = vel_normal * plane.normal.y + tangent_vel_y * friction_factor;
                        sphere.vel.z = vel_normal * plane.normal.z + tangent_vel_z * friction_factor;
                    }
                    
                    // Apply rolling resistance for spheres
                    let rolling_resistance = 0.01; // Small rolling resistance coefficient
                    let total_speed = (sphere.vel.x * sphere.vel.x + 
                                     sphere.vel.y * sphere.vel.y + 
                                     sphere.vel.z * sphere.vel.z).sqrt();
                    
                    if total_speed > 1e-3 {
                        let rolling_damping = (1.0 - rolling_resistance * dt * 60.0).max(0.0);
                        sphere.vel.x *= rolling_damping;
                        sphere.vel.y *= rolling_damping;
                        sphere.vel.z *= rolling_damping;
                    } else {
                        // Bring to complete stop for very slow motion
                        sphere.vel.x = 0.0;
                        sphere.vel.y = 0.0;
                        sphere.vel.z = 0.0;
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

        // Update spatial grid with current sphere positions
        self.spatial_grid.update(&self.spheres);
        
        // Get potential collision pairs from spatial grid (much faster than O(n²))
        let potential_pairs = self.spatial_grid.get_potential_pairs();
        
        // Sphere-sphere collision detection using spatial grid
        for (i, j) in potential_pairs {
            if i >= self.spheres.len() || j >= self.spheres.len() {
                continue; // Safety check
            }
            
            let dx = self.spheres[j].pos.x - self.spheres[i].pos.x;
            let dy = self.spheres[j].pos.y - self.spheres[i].pos.y;
            let dz = self.spheres[j].pos.z - self.spheres[i].pos.z;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            
            let radius1 = self.spheres[i].radius;
            let radius2 = self.spheres[j].radius;
            let min_distance = radius1 + radius2;
            
            if distance < min_distance && distance > 0.0 {
                // Collision detected, resolve it
                let overlap = min_distance - distance;
                let nx = dx / distance;
                let ny = dy / distance;
                let nz = dz / distance;
                
                // Get masses for collision response
                let mass1 = self.spheres[i].mass;
                let mass2 = self.spheres[j].mass;
                
                // Separate spheres based on position (more stable separation)
                // Use mass-weighted separation to avoid lighter objects moving too much
                let total_mass = mass1 + mass2;
                let sep_ratio_i = mass2 / total_mass; // Heavier objects move less
                let sep_ratio_j = mass1 / total_mass;
                
                self.spheres[i].pos.x -= nx * overlap * sep_ratio_i;
                self.spheres[i].pos.y -= ny * overlap * sep_ratio_i;
                self.spheres[i].pos.z -= nz * overlap * sep_ratio_i;
                
                self.spheres[j].pos.x += nx * overlap * sep_ratio_j;
                self.spheres[j].pos.y += ny * overlap * sep_ratio_j;
                self.spheres[j].pos.z += nz * overlap * sep_ratio_j;
                
                // Get material properties for collision response
                let mat1 = self.spheres[i].material;
                let mat2 = self.spheres[j].material;
                
                // Combined restitution (geometric mean for stability)
                let restitution = (mat1.restitution * mat2.restitution).sqrt();
                
                // Get relative velocity at contact point
                let rel_vel_x = self.spheres[j].vel.x - self.spheres[i].vel.x;
                let rel_vel_y = self.spheres[j].vel.y - self.spheres[i].vel.y;
                let rel_vel_z = self.spheres[j].vel.z - self.spheres[i].vel.z;
                
                // Relative velocity along collision normal
                let rel_vel_normal = rel_vel_x * nx + rel_vel_y * ny + rel_vel_z * nz;
                
                // Don't resolve if objects are separating
                if rel_vel_normal > 0.0 {
                    continue;
                }
                
                // Calculate inverse mass sum for impulse calculation
                let inv_mass_sum = 1.0 / mass1 + 1.0 / mass2;
                
                // Calculate impulse magnitude using proper physics formula
                // J = -(1 + e) * v_rel_n / (1/m1 + 1/m2)
                let impulse_magnitude = -(1.0 + restitution) * rel_vel_normal / inv_mass_sum;
                
                // Apply impulse to both spheres (impulse = change in momentum)
                let impulse_x = impulse_magnitude * nx;
                let impulse_y = impulse_magnitude * ny;
                let impulse_z = impulse_magnitude * nz;
                
                // Δv = J / m
                self.spheres[i].vel.x -= impulse_x / mass1;
                self.spheres[i].vel.y -= impulse_y / mass1;
                self.spheres[i].vel.z -= impulse_z / mass1;
                
                self.spheres[j].vel.x += impulse_x / mass2;
                self.spheres[j].vel.y += impulse_y / mass2;
                self.spheres[j].vel.z += impulse_z / mass2;
                
                // Apply friction (simplified Coulomb friction)
                let friction = (mat1.friction * mat2.friction).sqrt();
                
                // Tangential velocity (remove normal component)
                let tangent_vel_x = rel_vel_x - rel_vel_normal * nx;
                let tangent_vel_y = rel_vel_y - rel_vel_normal * ny;
                let tangent_vel_z = rel_vel_z - rel_vel_normal * nz;
                
                let tangent_speed = (tangent_vel_x * tangent_vel_x + 
                                   tangent_vel_y * tangent_vel_y + 
                                   tangent_vel_z * tangent_vel_z).sqrt();
                
                if tangent_speed > 1e-6 {
                    // Normalize tangent direction
                    let tangent_x = tangent_vel_x / tangent_speed;
                    let tangent_y = tangent_vel_y / tangent_speed;
                    let tangent_z = tangent_vel_z / tangent_speed;
                    
                    // Friction impulse (limited by Coulomb friction law)
                    let friction_impulse = friction * impulse_magnitude.abs();
                    
                    // Calculate maximum possible friction based on reduced mass
                    let reduced_mass = (mass1 * mass2) / (mass1 + mass2);
                    let max_friction = tangent_speed * reduced_mass;
                    let friction_magnitude = friction_impulse.min(max_friction);
                    
                    // Apply friction impulse
                    let friction_x = friction_magnitude * tangent_x;
                    let friction_y = friction_magnitude * tangent_y;
                    let friction_z = friction_magnitude * tangent_z;
                    
                    self.spheres[i].vel.x += friction_x / mass1;
                    self.spheres[i].vel.y += friction_y / mass1;
                    self.spheres[i].vel.z += friction_z / mass1;
                    
                    self.spheres[j].vel.x -= friction_x / mass2;
                    self.spheres[j].vel.y -= friction_y / mass2;
                    self.spheres[j].vel.z -= friction_z / mass2;
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
