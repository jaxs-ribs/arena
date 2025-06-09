#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
//! # JAXS Physics Engine
//!
//! A minimal, differentiable physics engine for the JAXS project (Just Another Exploration Substrate).
//!
//! This crate provides the foundational physics layer for the JAXS project. It
//! features a small set of rigid body types and a GPU-accelerated simulation
//! loop that is capable of backpropagation. The engine is designed to be
//! lightweight and efficient, serving as a core component for higher-level
//! machine learning tasks.
//!
//! ## Key Components
//!
//! -   **Rigid Bodies:** The engine supports several types of rigid bodies,
//!     including [`Sphere`], [`BoxBody`], [`Cylinder`], and [`Plane`]. These
//!     are defined in the [`types`] module.
//! -   **Simulation:** The [`PhysicsSim`] struct in the [`simulation`] module
//!     is the main entry point for running the physics simulation. It manages
//!     the state of all rigid bodies and steps the simulation forward in time.
//! -   **Differentiability:** A key feature of the JAXS physics engine is its
//!     support for differentiability, which allows for gradient-based
//!     optimization of physical parameters. This is crucial for the ML
//!     components of the project.
//!
//! ## Usage
//!
//! To use the physics engine, you typically start by creating an instance of
//! [`PhysicsSim`] and adding rigid bodies to it. You can then run the
//! simulation by calling the `run` method, specifying the time step and the
//! number of substeps.
//!
//! ```rust,ignore
//! use jaxs_physics::{PhysicsSim, Sphere};
//!
//! let mut sim = PhysicsSim::new();
//! let sphere = Sphere::new(1.0);
//! sim.add_sphere(sphere);
//!
//! let dt = 0.01;
//! let num_steps = 100;
//! sim.run(dt, num_steps)?;
//! ```

pub mod types;
pub mod simulation;

pub use simulation::{PhysicsError, PhysicsSim, SphereState};
pub use types::{
    BoxBody, BoundingBox, ContactDebugInfo, Cylinder, ForceDebugInfo, Joint, JointParams, 
    Material, PhysicsDebugInfo, PhysParams, Plane, Sphere, SpatialGrid, SpatialGridDebugInfo, 
    Vec3, VelocityDebugInfo,
};
