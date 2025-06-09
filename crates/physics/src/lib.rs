#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
//! Minimal differentiable physics engine.
//!
//! This crate defines a small set of rigid body types and a GPU accelerated
//! simulation loop capable of backpropagation. It is intentionally lightweight
//! and serves as the physics layer for the higher level ML components.

pub mod types;
pub mod simulation;

pub use simulation::{PhysicsError, PhysicsSim, SphereState};
pub use types::{
    BoxBody, Cylinder, Joint, JointParams, PhysParams, Plane, Sphere, Vec3,
};
