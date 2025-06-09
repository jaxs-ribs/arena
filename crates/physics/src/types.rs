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
/// Dynamic spherical rigid body.
///
/// A [`Sphere`] is represented by its position, velocity and orientation.
/// Padding fields are present so the structure matches the memory layout
/// expected by the GPU compute kernels.
pub struct Sphere {
    /// Center of mass position in world space.
    pub pos: Vec3,
    _pad1: f32,
    /// Linear velocity in metres per second.
    pub vel: Vec3,
    _pad2: f32,
    /// Orientation quaternion stored as `[x, y, z, w]`.
    pub orientation: [f32; 4],
    /// Angular velocity in radians per second.
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
/// Global simulation parameters.
///
/// These values control how bodies are integrated each step. `gravity`
/// specifies the acceleration applied to all dynamic objects. `dt` is the time
/// step used by [`crate::simulation::PhysicsSim`] and `forces` holds per-sphere external forces on
/// the X and Y axes.
pub struct PhysParams {
    /// Gravity vector applied to all dynamic bodies.
    pub gravity: Vec3,
    /// Simulation time step.
    pub dt: f32,
    /// External forces applied per sphere (x/y components). Length must match number of spheres.
    pub forces: Vec<[f32; 2]>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Constraint linking two bodies together at a fixed distance.
pub struct Joint {
    /// Index of the first body in [`crate::simulation::PhysicsSim::spheres`].
    pub body_a: u32,
    /// Index of the second body.
    pub body_b: u32,
    /// Distance the joint attempts to maintain between the two bodies.
    pub rest_length: f32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Global parameters controlling joint solver behaviour.
pub struct JointParams {
    /// Compliance for position based dynamics joints. Higher values allow
    /// greater stretching.
    pub compliance: f32,
    pub _pad: [f32; 3],
}

#[derive(Copy, Clone, Debug)]
/// Axis aligned dynamic box used for simple collision tests.
pub struct BoxBody {
    /// Center of mass position.
    pub pos: Vec3,
    /// Half extents defining the box dimensions.
    pub half_extents: Vec3,
    /// Linear velocity in metres per second.
    pub vel: Vec3,
}

#[derive(Copy, Clone, Debug)]
/// Dynamic cylinder primitive.
pub struct Cylinder {
    /// Center of mass position.
    pub pos: Vec3,
    /// Linear velocity.
    pub vel: Vec3,
    /// Radius of the circular base.
    pub radius: f32,
    /// Height of the cylinder.
    pub height: f32,
}

#[derive(Copy, Clone, Debug)]
/// Infinite plane used as a static collision primitive.
pub struct Plane {
    /// Plane normal should be normalized.
    pub normal: Vec3,
    /// Plane equation: `normal.dot(x) + d = 0`.
    pub d: f32,
}
