#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod types;
pub mod simulation;

pub use simulation::{PhysicsError, PhysicsSim, SphereState};
pub use types::{
    BoxBody, Cylinder, Joint, JointParams, PhysParams, Plane, Sphere, Vec3,
};
