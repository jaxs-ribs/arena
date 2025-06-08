# Differentiable-Physics

This project aims to build a full stack physics and machine learning environment inspired by [Brax](https://github.com/google/brax) but designed around **WebGPU** and implemented entirely in Rust.  The goal is to provide fast differentiable physics, a compute backend that works on desktop and in browsers, and a minimal runtime for training ML policies directly against the simulator.

## Vision

* **WebGPU First** – leverage the modern graphics and compute API available across platforms. The `compute` crate provides a generic interface over GPU compute kernels with a CPU fallback used during testing. The initial GPU implementation targets the `wgpu` crate so it can run on Vulkan, Metal and eventually WebGPU in the browser.
* **Physics Engine** – the `physics` crate contains a simple rigid body simulator implemented with GPU kernels. At the moment it includes a basic sphere integrator but it is structured so more complex bodies and constraints can be added. The API is designed to be differentiable so gradients can flow through simulation steps.
* **Runtime & Tooling** – the `runtime` crate is a small executable that wires everything together. It includes a shader watcher for hot‑reloading WGSL compute shaders and serves as the basis for training loops and visual debugging.
* **ML Policy Stack** – future work will extend the runtime with reinforcement learning utilities and policy networks so the simulator can be used end‑to‑end for control tasks. Everything stays in Rust for portability and performance.

## Repository Layout

```
crates/
  compute/   # Abstraction over compute backends (CPU mock, WGPU)
  physics/   # Differentiable physics simulation code
  runtime/   # Executable with shader hot‑reload and training scaffolding
shaders/     # WGSL compute kernels
tests/       # Integration tests (e.g. free‑fall verification)
```

Each crate has its own `Cargo.toml` and unit tests. The workspace root defines common dependencies and ensures everything builds together.

## Running Tests

The project is developed on stable Rust. Clone the repository, then run:

```bash
rustup override set stable
cargo test
```

Unit tests cover the CPU compute backend, physics stepping, shader compilation and a simple integration test verifying that a falling sphere matches analytic physics. More GPU‑specific tests will be added as the WebGPU backend matures.

## Command Cheatsheet

Below is a quick reference of commands for running the various test suites and examples. All commands assume you are at the repository root.

```bash
# Run the entire test suite (CPU fallback backend)
cargo test

# Run only compute crate tests
cargo test -p compute

# Run physics crate tests
cargo test -p physics

# Execute integration tests under `tests/`
cargo test --test free_fall

# Compile and run GPU kernels on macOS with Metal
cargo test -p compute --features metal

# Compile benches (requires `criterion`)
cargo bench

# Launch the renderer example
cargo run -p runtime --features render -- --draw
```

## Visualizing the simulation

The project includes a minimal renderer based on `wgpu`. To see a live sphere
falling under gravity, build the runtime with the `render` feature and pass the
`--draw` flag to the binary:

```bash
cargo run -p runtime --features render -- --draw
```

This will open a window and draw the sphere positions after each simulation
step.

## JSON Schema

The file [docs/README.md](docs/README.md) describes the JSON format used to
construct creature bodies.

## Status

The codebase is early and experimental. Right now it demonstrates the core pieces needed to integrate physics simulation with GPU kernels compiled from WGSL. The intention is to evolve this into a Brax‑like environment where reinforcement learning policies can be trained directly on a differentiable WebGPU simulator.

Contributions and issue reports are welcome as we iterate on the design and expand the feature set.
