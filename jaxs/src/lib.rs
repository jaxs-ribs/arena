//! # JAXS Project
//!
//! Welcome to the documentation for the JAXS project.
//!
//! JAXS is a differentiable physics engine and reinforcement learning framework
//! for creating and training simulated creatures.
//!
//! ## Project Structure
//!
//! The project is organized into several crates, each with a specific responsibility:
//!
//! -   **runtime**: The main entry point of the application. It ties together all the other crates and runs the main simulation loop.
//! - [`physics`]: A minimal, differentiable physics engine for rigid body simulation.
//! - [`render`]: A simple `wgpu`-based rendering engine for visualizing physics simulations.
//! - [`compute`]: A unified CPU and GPU compute abstraction layer.
//! - [`ml`]: Contains the machine learning components, including neural networks and reinforcement learning algorithms.
//! - [`phenotype`]: Defines the structure and behavior of the simulated creatures.
//!
//! Start by exploring the `runtime` crate's documentation to understand how the application is structured and run.

pub use compute;
pub use ml;
pub use phenotype;
pub use physics;
pub use render; 