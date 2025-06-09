//! # JAXS: Just Another Exploration Substrate
//!
//! Welcome to the documentation for JAXS (Just Another Exploration Substrate).
//!
//! ## Overview
//!
//! JAXS is a full-stack physics and machine learning environment inspired by
//! [Brax](https://github.com/google/brax), but designed around **WebGPU** and
//! implemented entirely in Rust. The goal is to provide fast, differentiable
//! physics, a compute backend that works on both desktop and in the browser,
//! and a minimal runtime for training machine learning policies directly
//! against the simulator.
//!
//! ## Core Philosophy
//!
//! The project is built on two key principles:
//!
//! -   **WebGPU First:** JAXS leverages the modern graphics and compute API of
//!     WebGPU to ensure portability and performance across a wide range of
//!     platforms. The [`compute`] crate provides a generic interface over GPU
//!     compute kernels, with a CPU fallback for testing purposes.
//! -   **End-to-End Rust:** By using Rust for the entire stack, from the
//!     physics engine to the machine learning policies, JAXS ensures memory
//!     safety, high performance, and a seamless development experience.
//!
//! ## Project Architecture
//!
//! JAXS is organized into a series of interconnected crates, each with a
//! distinct responsibility. This modular architecture allows for clear
//! separation of concerns and makes the project easier to understand,
//! maintain, and extend.
//!
//! ### The Crates
//!
//! -   **`jaxs`:** The crate you are currently viewing. It serves as the main
//!     entry point for both the documentation and the executable application. It
//!     ties together all the other crates and runs the main simulation loop.
//! -   **[`compute`]:** A thin abstraction layer over different compute
//!     backends. It provides a mock CPU backend for testing and a `wgpu`-based
//!     backend for production.
//! -   **[`physics`]:** A differentiable rigid body physics engine. It is
//!     designed to be extensible, with support for various constraints and
//!     articulated structures.
//! -   **[`ml`]:** Contains all the building blocks for machine learning,
//!     including tensor operations, automatic differentiation, and reinforcement
//!     learning utilities.
//! -   **[`phenotype`]:** Defines the structure and behavior of the simulated
//!     creatures, allowing for procedural generation and evolution.
//!
//! ## Long-Term Vision
//!
//! The ultimate goal of JAXS is to explore rich artificial life through a
//! combination of physics simulation, reinforcement learning, and evolutionary
//! search. The long-term roadmap includes:
//!
//! -   **Character Creation:** A JSON-driven format for defining bodies,
//!     joints, and actuators, allowing for the procedural generation and
//!     evolution of agents.
//! -   **Reinforcement Learning:** The implementation of advanced algorithms
//!     like Dreamer V3, with a shared world model across all agents.
//! -   **Novelty-Driven Search:** Inspired by Kenneth Stanley's work on
//!     automated search, JAXS will use the latent vector of the world model to
//!     measure behavioral novelty, guiding exploration towards truly unique
//!     behaviors.
//!
//! ## Getting Started
//!
//! To get started with JAXS, it is recommended to explore the documentation for
//! each crate, starting with the [`jaxs`] crate to understand how the
//! application is launched and managed. From there, you can dive into the
//! [`physics`] and [`ml`] crates to understand the core mechanics of the
//! simulation and learning processes.

pub use compute;
pub use ml;
pub use phenotype;
pub use physics;
pub use render; 