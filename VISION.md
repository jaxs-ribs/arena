# JAXS Vision

JAXS (Just Another Exploration Substrate) aims to build a full stack differentiable physics and machine learning environment with a design inspired by [Brax](https://github.com/google/brax). Everything is implemented in **Rust** and targets **WebGPU** so that experiments can run efficiently on desktop GPUs and eventually in the browser. The long term goal is to explore rich artificial life through a combination of physics simulation, reinforcement learning, and evolutionary search.

## Goals

* **Differentiable Physics Engine** – a GPU accelerated rigid body simulator that exposes gradients so that policies and model parameters can be optimized end to end. Simulation kernels will run via WebGPU to enable portability across platforms.
* **Character Creation** – a JSON driven format for defining bodies, joints, and actuators. Higher level genotypes expand into phenotypes so that agents can be procedurally generated or evolved.
* **Reinforcement Learning** – utilities for training policies directly against the simulator. The codebase will implement the Dreamer V3 algorithm with a shared world model across agents.
* **Novelty Driven Search** – drawing inspiration from Kenneth Stanley’s work on automated search for artificial life, the latent vector of the world model will be used to measure behavioral novelty. Evolutionary techniques and gradient based optimization will work together to discover new phenotypes.

## Why Rust and WebGPU?

Rust provides strong safety guarantees, reliable tooling, and a vibrant ecosystem for systems programming. WebGPU offers a modern compute API that runs on Vulkan, Metal, and DirectX, and will soon be available in all major browsers. This combination allows the engine to run both natively and in web environments while keeping a single codebase.

## High Level Architecture

The project is organized into several crates:

* `compute` – a thin abstraction over compute backends. A mock CPU backend is used during testing while the production backend uses `wgpu`.
* `physics` – differentiable rigid body simulation with an extensible solver. Current work focuses on simple primitives and contact resolution but the architecture is built for future constraints and articulated structures.
* `ml` – tensor operations, automatic differentiation, and reinforcement learning utilities.
* `render` – optional visualization of environments using WebGPU.
* `runtime` – a small executable that glues everything together. It includes shader hot reloading and will serve as the base for training loops.

## Novelty Search with Dreamer V3

Dreamer V3 learns a predictive world model and optimizes policies in the learned latent space. Our approach extends this by feeding the latent embeddings into a novelty module inspired by the paper **Automated Search for Artificial Life**. The novelty module scores how unique an agent’s behavior is compared to a growing archive. Gradients from this score propagate into the policy and world model so that exploration is directed toward truly novel behaviors. In addition, evolutionary strategies mutate genotypes to introduce structural diversity. The ultimate goal is an open‑ended search for interesting virtual creatures.

## Status and Roadmap

The repository currently provides the scaffolding for compute kernels, a basic physics engine, and initial reinforcement learning environments such as stick balancing. The next milestones are:

1. Expand the physics engine with more constraints and articulated bodies.
2. Integrate a minimal Dreamer V3 implementation using the `ml` crate.
3. Implement novelty measurement in the latent space and evolve genotypes accordingly.
4. Provide a WebGPU renderer and browser runner so experiments can be visualized online.

This is a research oriented project and many components are experimental. Community contributions, issue reports, and discussion are welcome as we iterate toward a fully featured platform for artificial life research.
