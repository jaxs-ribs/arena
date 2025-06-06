#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod simulation;
pub mod steps;
pub mod types;

pub use error::PhysicsError;
pub use simulation::PhysicsSim;
pub use types::{Joint, JointParams, PhysParams, Sphere, SphereState, Vec3};
