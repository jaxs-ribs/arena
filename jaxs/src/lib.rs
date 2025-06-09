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
//! The design of JAXS is guided by a few core principles:
//!
//! -   **Portability First:** The entire project is built to be portable. By
//!     targeting WebGPU, the physics and machine learning computations can run
//!     on a wide range of hardware (Vulkan, Metal, DirectX) and platforms,
//!     including native desktops and web browsers, all from a single codebase.
//! -   **Differentiability is Key:** Every component in the simulation pipeline,
//!     especially the physics engine, is designed to be differentiable. This
//!     allows for gradient-based optimization of not just ML policies, but also
//!     physical parameters, which is crucial for many advanced ML techniques.
//! -   **End-to-End Rust:** By using Rust for the entire stack, from the
//!     low-level physics to the high-level learning algorithms, JAXS ensures
//!     memory safety, high performance, and a seamless, unified development
//!     experience.
//!
//! ## The JAXS Architecture
//!
//! JAXS is organized into a series of interconnected crates, each with a
//! distinct responsibility. This modular architecture allows for clear
//! separation of concerns and makes the project easier to understand,
//! maintain, and extend.
//!
//! ### The Compute Model: A Fixed Opset
//!
//! At the heart of JAXS is the [`compute`] crate, which provides an abstraction
//! over CPU and GPU execution. Unlike general-purpose tensor libraries, JAXS
//! uses a fixed set of operations, or an "opset," defined in the
//! `compute::Kernel` enum.
//!
//! This design has several advantages:
//!
//! -   **Guaranteed Portability:** Every backend is only required to implement
//!     this specific set of kernels. This makes it straightforward to add new
//!     backends and guarantees that any computation will run on any supported
//!     platform.
//! -   **Performance:** Kernels can be hand-optimized for specific hardware,
//!     ensuring maximum performance for the operations that are most critical
//!     to the physics and ML workloads.
//! -   **Simplicity:** It avoids the complexity of a full-blown computation
//!     graph and kernel fusion system, keeping the core of the engine lean and
//!     focused.
//!
//! ### CPU/GPU Branching
//!
//! The `compute::ComputeBackend` trait is the key to JAXS's platform
//! flexibility. The project provides two primary implementations:
//!
//! -   **`CpuBackend`:** A reference implementation written in pure Rust. It is
//!     used for testing, debugging, and as a fallback on systems without a
//!     compatible GPU.
//! -   **`WgpuBackend`:** A high-performance implementation that uses the `wgpu`
//!     crate to execute compute shaders written in WGSL.
//!
//! The switch between these backends is handled at compile time via the `gpu`
//! feature flag. When enabled, the `default_backend()` function will provide
//! the `WgpuBackend`, seamlessly accelerating the simulation.
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
//! each crate, starting with the `jaxs` crate to understand how the
//! application is launched and managed. From there, you can dive into the
//! [`physics`] and [`ml`] crates to understand the core mechanics of the
//! simulation and learning processes.

pub use compute;
pub use ml;
pub use phenotype;
pub use physics;
#[cfg(feature = "render")]
pub use render; 