#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod types;
pub mod simulation;

pub use simulation::{PhysicsError, PhysicsSim, SphereState};
pub use types::{Joint, JointParams, PhysParams, Sphere, Vec3};
