//! # Physics Data Types
//!
//! This module defines the core data structures used throughout the JAXS physics
//! engine. These types are designed to be efficient and compatible with the
//! GPU-accelerated simulation loop.
//!
//! ## Overview
//!
//! The data structures in this module can be categorized as follows:
//!
//! -   **Geometric Primitives:** These are the basic building blocks for rigid
//!     bodies, such as [`Vec3`] for positions and velocities.
//! -   **Rigid Bodies:** These represent the dynamic objects in the simulation,
//!     including [`Sphere`], [`BoxBody`], and [`Cylinder`].
//! -   **Constraints:** These are used to connect rigid bodies, such as the
//!     [`Joint`] struct.
//! -   **Simulation Parameters:** These control the global behavior of the
//!     physics simulation, such as [`PhysParams`] and [`JointParams`].
//!
//! ## GPU Compatibility
//!
//! Many of the data structures in this module are marked with `#[repr(C)]` and
//! derive the [`bytemuck::Pod`] and [`bytemuck::Zeroable`] traits. This ensures
//! that their memory layout is compatible with the GPU, allowing them to be
//! transferred to and from the GPU without any conversion. This is a key
//! feature of the JAXS physics engine, as it enables the use of GPU
//! acceleration for the simulation loop.

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Three dimensional vector used by the physics engine.
///
/// This simple data structure is shared by all bodies to represent
/// positions, velocities and directions. It is marked as [`bytemuck::Pod`] so it can be
/// transferred to the GPU without conversion.
pub struct Vec3 {
    /// X component of the vector.
    pub x: f32,
    /// Y component of the vector.
    pub y: f32,
    /// Z component of the vector.
    pub z: f32,
}

impl Vec3 {
    /// Creates a new [`Vec3`] with the provided components.
    ///
    /// This constructor is `const` so that vectors can be used in
    /// constant expressions when building static geometry or parameters.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// A dynamic spherical rigid body.
///
/// A `Sphere` is one of the fundamental rigid body types in the JAXS physics
/// engine. It is represented by its position, velocity, orientation, and
/// angular velocity.
///
/// The struct is designed to be directly mappable to GPU memory, hence the
/// padding fields to ensure proper alignment. This allows for efficient
/// transfer of sphere data to the GPU for accelerated physics calculations.
pub struct Sphere {
    /// The world-space position of the sphere's center of mass.
    pub pos: Vec3,
    _pad1: f32,
    /// The linear velocity of the sphere, measured in meters per second.
    pub vel: Vec3,
    _pad2: f32,
    /// The orientation of the sphere, represented as a quaternion in `[x, y, z, w]`
    /// format.
    pub orientation: [f32; 4],
    /// The angular velocity of the sphere, measured in radians per second.
    pub angular_vel: Vec3,
    _pad3: f32,
}

impl Sphere {
    /// Constructs a new [`Sphere`] at the given position and velocity.
    ///
    /// The orientation is initialised to the identity quaternion and
    /// angular velocity is zero.
    #[must_use]
    pub const fn new(pos: Vec3, vel: Vec3) -> Self {
        Self {
            pos,
            _pad1: 0.0,
            vel,
            _pad2: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::new(0.0, 0.0, 0.0),
            _pad3: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Axis aligned box used by some contact generation routines.
///
/// This type mirrors the memory layout expected by compute shaders. It is not
/// currently exposed publicly but is used internally when dispatching kernels
/// that operate on [`BoxBody`] instances.
pub struct BoxShape {
    /// Center of the box in world space.
    pub center: Vec3,
    _pad1: f32,
    /// Half extents along each axis.
    pub half_extents: Vec3,
    _pad2: f32,
}

#[derive(Clone, Debug)]
/// Global parameters that control the physics simulation.
///
/// These values influence the behavior of all rigid bodies in the simulation.
/// They are typically set once at the beginning of the simulation.
pub struct PhysParams {
    /// The gravitational acceleration applied to all dynamic objects.
    pub gravity: Vec3,
    /// The time step for each simulation update, used by [`crate::simulation::PhysicsSim`].
    pub dt: f32,
    /// A list of external forces to be applied to each sphere on the X and Y axes.
    /// The length of this vector must match the number of spheres in the simulation.
    pub forces: Vec<[f32; 2]>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Constraint linking two bodies together at a fixed distance.
pub struct Joint {
    /// Index of the first body in [`crate::simulation::PhysicsSim::spheres`].
    pub body_a: u32,
    /// Index of the second body in [`crate::simulation::PhysicsSim::spheres`].
    pub body_b: u32,
    /// The target distance that the joint tries to maintain between the two bodies.
    pub rest_length: f32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Global parameters that control the behavior of the joint solver.
pub struct JointParams {
    /// The compliance of the position-based dynamics (PBD) joints. A higher
    /// value results in more "stretchy" joints.
    pub compliance: f32,
    pub _pad: [f32; 3],
}

#[derive(Copy, Clone, Debug)]
/// A dynamic, axis-aligned bounding box (AABB) used for simplified
/// collision detection.
pub struct BoxBody {
    /// The position of the center of mass.
    pub pos: Vec3,
    /// The half-extents of the box, defining its dimensions along each axis.
    pub half_extents: Vec3,
    /// The linear velocity of the box, measured in meters per second.
    pub vel: Vec3,
}

#[derive(Copy, Clone, Debug)]
/// A dynamic cylinder primitive.
pub struct Cylinder {
    /// The position of the center of mass.
    pub pos: Vec3,
    /// The linear velocity of the cylinder.
    pub vel: Vec3,
    /// The radius of the cylinder's circular base.
    pub radius: f32,
    /// The height of the cylinder.
    pub height: f32,
}

#[derive(Copy, Clone, Debug)]
/// An infinite plane used as a static collision primitive.
///
/// A plane is defined by its normal vector and its distance from the origin.
/// It is typically used as a static ground or wall in the simulation.
pub struct Plane {
    /// The normal vector of the plane. This vector should be normalized to
    /// have a length of 1.
    pub normal: Vec3,
    /// The distance of the plane from the origin, along its normal vector.
    /// The plane equation is `normal.dot(x) + d = 0`.
    pub d: f32,
}
