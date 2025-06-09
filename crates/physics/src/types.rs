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

    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn normalize(&self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            Self::ZERO
        }
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(val: Vec3) -> Self {
        [val.x, val.y, val.z]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Material properties for physical interactions.
///
/// Defines surface properties that control how objects interact during collisions.
pub struct Material {
    /// Friction coefficient. 0.0 = frictionless, 1.0 = high friction.
    pub friction: f32,
    /// Restitution coefficient. 0.0 = perfectly inelastic, 1.0 = perfectly elastic.
    pub restitution: f32,
    /// Padding for alignment.
    _pad: [f32; 2],
}

impl Material {
    /// Creates a new material with the specified friction and restitution.
    pub const fn new(friction: f32, restitution: f32) -> Self {
        Self {
            friction,
            restitution,
            _pad: [0.0; 2],
        }
    }

    /// Default material with moderate friction and some bounce.
    pub const fn default() -> Self {
        Self::new(0.5, 0.3)
    }

    /// Bouncy ball material.
    pub const fn bouncy() -> Self {
        Self::new(0.3, 0.9)
    }

    /// Ice-like material with low friction and some bounce.
    pub const fn slippery() -> Self {
        Self::new(0.05, 0.6)
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
    /// The radius of the sphere.
    pub radius: f32,
    /// The linear velocity of the sphere, measured in meters per second.
    pub vel: Vec3,
    /// The mass of the sphere in kilograms.
    pub mass: f32,
    /// The orientation of the sphere, represented as a quaternion in `[x, y, z, w]`
    /// format.
    pub orientation: [f32; 4],
    /// The angular velocity of the sphere, measured in radians per second.
    pub angular_vel: Vec3,
    _pad2: f32,
    /// Material properties for collision response.
    pub material: Material,
}

impl Sphere {
    /// Constructs a new [`Sphere`] at the given position and velocity with specified radius.
    ///
    /// The orientation is initialised to the identity quaternion and
    /// angular velocity is zero. Uses default material properties and unit mass.
    #[must_use]
    pub const fn new(pos: Vec3, vel: Vec3, radius: f32) -> Self {
        Self {
            pos,
            radius,
            vel,
            mass: 1.0, // Default unit mass
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::new(0.0, 0.0, 0.0),
            _pad2: 0.0,
            material: Material::default(),
        }
    }

    /// Constructs a new [`Sphere`] with custom material properties.
    #[must_use]
    pub const fn with_material(pos: Vec3, vel: Vec3, radius: f32, material: Material) -> Self {
        Self {
            pos,
            radius,
            vel,
            mass: 1.0, // Default unit mass
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::new(0.0, 0.0, 0.0),
            _pad2: 0.0,
            material,
        }
    }

    /// Constructs a new [`Sphere`] with custom mass and material properties.
    #[must_use]
    pub const fn with_mass_and_material(pos: Vec3, vel: Vec3, radius: f32, mass: f32, material: Material) -> Self {
        Self {
            pos,
            radius,
            vel,
            mass,
            orientation: [0.0, 0.0, 0.0, 1.0],
            angular_vel: Vec3::new(0.0, 0.0, 0.0),
            _pad2: 0.0,
            material,
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
/// A hinge joint allowing rotation around a single axis.
pub struct RevoluteJoint {
    /// Index of the first body.
    pub body_a: u32,
    /// Index of the second body.
    pub body_b: u32,
    /// Anchor point on body A in local coordinates.
    pub anchor_a: Vec3,
    /// Anchor point on body B in local coordinates.
    pub anchor_b: Vec3,
    /// Rotation axis in world coordinates.
    pub axis: Vec3,
    /// Lower angular limit in radians.
    pub lower_limit: f32,
    /// Upper angular limit in radians.
    pub upper_limit: f32,
    /// Target motor speed in radians per second.
    pub motor_speed: f32,
    /// Maximum motor force.
    pub motor_max_force: f32,
    /// Enable motor when non-zero.
    pub enable_motor: u32,
    /// Enable limits when non-zero.
    pub enable_limit: u32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// A sliding joint constraining motion along an axis.
pub struct PrismaticJoint {
    pub body_a: u32,
    pub body_b: u32,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub lower_limit: f32,
    pub upper_limit: f32,
    pub motor_speed: f32,
    pub motor_max_force: f32,
    pub enable_motor: u32,
    pub enable_limit: u32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// A ball-and-socket joint allowing 3 DoF rotation.
pub struct BallJoint {
    pub body_a: u32,
    pub body_b: u32,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// A rigid joint locking two bodies together.
pub struct FixedJoint {
    pub body_a: u32,
    pub body_b: u32,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// Relative rotation stored as a quaternion.
    pub relative_rotation: [f32; 4],
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
    /// Material properties for collision response.
    pub material: Material,
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
    /// Material properties for collision response.
    pub material: Material,
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

/// A uniform spatial grid for broad-phase collision detection.
///
/// This grid subdivides 3D space into cubic cells and maintains lists of
/// objects in each cell. This allows for efficient pruning of collision
/// pairs by only checking objects in the same or neighboring cells.
pub struct SpatialGrid {
    /// Size of each grid cell (same in all dimensions).
    pub cell_size: f32,
    /// Grid bounds - objects outside these bounds are not tracked.
    pub bounds: BoundingBox,
    /// Dimensions of the grid (number of cells in each direction).
    pub dimensions: [usize; 3],
    /// Flattened array of cells, each containing a list of sphere indices.
    pub cells: Vec<Vec<usize>>,
}

/// Axis-aligned bounding box used for spatial grid bounds.
#[derive(Copy, Clone, Debug)]
pub struct BoundingBox {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
}

/// Debug information for visualizing physics state.
#[derive(Clone, Debug)]
pub struct PhysicsDebugInfo {
    /// Contact points between objects.
    pub contacts: Vec<ContactDebugInfo>,
    /// Velocity vectors for all spheres.
    pub velocity_vectors: Vec<VelocityDebugInfo>,
    /// Force vectors applied to spheres.
    pub force_vectors: Vec<ForceDebugInfo>,
    /// Spatial grid visualization data.
    pub spatial_grid_info: SpatialGridDebugInfo,
}

/// Debug information for a contact point.
#[derive(Copy, Clone, Debug)]
pub struct ContactDebugInfo {
    /// Position of the contact point.
    pub position: Vec3,
    /// Normal vector at the contact point.
    pub normal: Vec3,
    /// Penetration depth.
    pub depth: f32,
    /// Indices of the objects in contact.
    pub object_indices: (usize, usize),
}

/// Debug information for visualizing velocity.
#[derive(Copy, Clone, Debug)]
pub struct VelocityDebugInfo {
    /// Starting position of the velocity vector.
    pub position: Vec3,
    /// Velocity vector.
    pub velocity: Vec3,
    /// Index of the object.
    pub object_index: usize,
}

/// Debug information for visualizing forces.
#[derive(Copy, Clone, Debug)]
pub struct ForceDebugInfo {
    /// Starting position of the force vector.
    pub position: Vec3,
    /// Force vector.
    pub force: Vec3,
    /// Index of the object.
    pub object_index: usize,
}

/// Debug information for spatial grid visualization.
#[derive(Clone, Debug)]
pub struct SpatialGridDebugInfo {
    /// Grid cell size.
    pub cell_size: f32,
    /// Grid bounds.
    pub bounds: BoundingBox,
    /// Grid dimensions.
    pub dimensions: [usize; 3],
    /// Occupied cells with their object counts.
    pub occupied_cells: Vec<(usize, usize)>, // (cell_index, object_count)
}

impl SpatialGrid {
    /// Creates a new spatial grid with the specified parameters.
    ///
    /// # Arguments
    /// * `cell_size` - Size of each grid cell (should be roughly 2x the average object radius)
    /// * `bounds` - World-space bounds that the grid covers
    pub fn new(cell_size: f32, bounds: BoundingBox) -> Self {
        let dimensions = [
            ((bounds.max.x - bounds.min.x) / cell_size).ceil() as usize,
            ((bounds.max.y - bounds.min.y) / cell_size).ceil() as usize,
            ((bounds.max.z - bounds.min.z) / cell_size).ceil() as usize,
        ];
        
        let total_cells = dimensions[0] * dimensions[1] * dimensions[2];
        let cells = vec![Vec::new(); total_cells];
        
        Self {
            cell_size,
            bounds,
            dimensions,
            cells,
        }
    }
    
    /// Converts a world position to grid coordinates.
    fn world_to_grid(&self, pos: Vec3) -> [i32; 3] {
        [
            ((pos.x - self.bounds.min.x) / self.cell_size).floor() as i32,
            ((pos.y - self.bounds.min.y) / self.cell_size).floor() as i32,
            ((pos.z - self.bounds.min.z) / self.cell_size).floor() as i32,
        ]
    }
    
    /// Converts grid coordinates to a flat cell index.
    fn grid_to_index(&self, grid_coords: [i32; 3]) -> Option<usize> {
        if grid_coords[0] < 0 || grid_coords[1] < 0 || grid_coords[2] < 0 ||
           grid_coords[0] >= self.dimensions[0] as i32 ||
           grid_coords[1] >= self.dimensions[1] as i32 ||
           grid_coords[2] >= self.dimensions[2] as i32 {
            return None;
        }
        
        let index = (grid_coords[2] as usize) * self.dimensions[0] * self.dimensions[1] +
                   (grid_coords[1] as usize) * self.dimensions[0] +
                   (grid_coords[0] as usize);
        Some(index)
    }
    
    /// Clears all cells and repopulates them with the given spheres.
    pub fn update(&mut self, spheres: &[Sphere]) {
        // Clear all cells
        for cell in &mut self.cells {
            cell.clear();
        }
        
        // Insert each sphere into appropriate cells
        for (i, sphere) in spheres.iter().enumerate() {
            self.insert_sphere(i, sphere);
        }
    }
    
    /// Inserts a sphere into the appropriate grid cells.
    /// A sphere may occupy multiple cells if it spans cell boundaries.
    fn insert_sphere(&mut self, sphere_index: usize, sphere: &Sphere) {
        let radius = sphere.radius;
        let pos = sphere.pos;
        
        // Find the range of cells this sphere occupies
        let min_grid = self.world_to_grid(Vec3::new(
            pos.x - radius, pos.y - radius, pos.z - radius
        ));
        let max_grid = self.world_to_grid(Vec3::new(
            pos.x + radius, pos.y + radius, pos.z + radius
        ));
        
        // Insert into all overlapping cells
        for z in min_grid[2]..=max_grid[2] {
            for y in min_grid[1]..=max_grid[1] {
                for x in min_grid[0]..=max_grid[0] {
                    if let Some(cell_index) = self.grid_to_index([x, y, z]) {
                        self.cells[cell_index].push(sphere_index);
                    }
                }
            }
        }
    }
    
    /// Returns potential collision pairs by checking overlapping grid cells.
    /// This dramatically reduces the number of collision checks needed.
    pub fn get_potential_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        
        for cell in &self.cells {
            // Check all pairs within this cell
            for i in 0..cell.len() {
                for j in (i + 1)..cell.len() {
                    pairs.push((cell[i], cell[j]));
                }
            }
        }
        
        // Remove duplicates (spheres can be in multiple cells)
        pairs.sort_unstable();
        pairs.dedup();
        
        pairs
    }
}
