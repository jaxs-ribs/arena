# JAXS: Just Another Exploration Substrate

JAXS aims to build a full stack physics and machine learning environment inspired by [Brax](https://github.com/google/brax) but designed around **WebGPU** and implemented entirely in Rust.  The goal is to provide fast differentiable physics, a compute backend that works on desktop and in browsers, and a minimal runtime for training ML policies directly against the simulator.

> *See [VISION.md](VISION.md) for the long-term roadmap, including novelty-driven search with Dreamer V3.*

## Vision

* **WebGPU First** – leverage the modern graphics and compute API available across platforms. The `compute` crate provides a generic interface over GPU compute kernels with a CPU fallback used during testing. The initial GPU implementation targets the `wgpu` crate so it can run on Vulkan, Metal and eventually WebGPU in the browser.
* **Physics Engine** – the `physics` crate contains a simple rigid body simulator implemented with GPU kernels. At the moment it includes a basic sphere integrator but it is structured so more complex bodies and constraints can be added. The API is designed to be differentiable so gradients can flow through simulation steps.
* **ML Policy Stack** – future work will extend the runtime with reinforcement learning utilities and policy networks so the simulator can be used end‑to‑end for control tasks. Everything stays in Rust for portability and performance.
* **Runtime & Tooling** – the `jaxs` crate is a small executable that wires everything together. It includes a shader watcher for hot‑reloading WGSL compute shaders and serves as the basis for training loops and visual debugging.

## Repository Layout

```
crates/
  compute/   # Abstraction over compute backends (CPU mock, WGPU)
  ml/        # ML building blocks (tensor, tape, optim)
  physics/   # Differentiable physics simulation code
  render/    # WGPU-based renderer for visualization
jaxs/        # Executable and documentation hub
shaders/     # WGSL compute kernels
tests/       # ML integration tests
```

Each crate has its own `Cargo.toml` and unit tests. The workspace root defines common dependencies and ensures everything builds together.

## Running Tests

The project is developed on stable Rust. Clone the repository, then run:

```bash
rustup override set stable
cargo test
```

Unit tests cover the CPU compute backend, physics stepping, shader compilation and a simple integration test verifying that a falling sphere matches analytic physics. GPU-specific tests are part of the standard `cargo test` run and do not require a separate command.

## Command Cheatsheet

Below is a quick reference of commands for running the various test suites and examples. All commands assume you are at the repository root.

```bash
# Run the entire test suite (CPU fallback backend)
cargo test

# Run only compute crate tests
cargo test -p compute

# Run physics crate tests
cargo test -p physics

# Execute a specific integration test from the `ml` crate
cargo test --test 01_ops

# Run compute crate tests with the WGPU backend
cargo test -p compute --features gpu

# Compile benches (requires `criterion`)
cargo bench

# Launch the renderer example
cargo run -p jaxs --features render -- --draw

# Build and view API documentation
cargo doc --workspace --no-deps && open target/doc/jaxs/index.html
```

## Visualizing the simulation

The project includes a renderer based on `wgpu` for visualizing physics simulations. 
By default, it runs the CartPole gym demo. To run with rendering:

```bash
cargo run --release --features render
```

This will open a window showing the CartPole environment with interactive controls.
The simulation runs at 60Hz with real-time physics updates.


## JSON Schema

The file [docs/README.md](docs/README.md) describes the JSON format used to
construct creature bodies.

## CartPole Environment

The repository includes a full CartPole reinforcement learning environment with:

- **Physics-based simulation**: Revolute joint constraints connecting carts and poles
- **2D motion constraints**: CartPoles are constrained to move in the X-Y plane only
- **Configurable parameters**: Cart mass, pole length/mass, failure angles, position limits
- **Grid layout**: Create multiple CartPoles for parallel training
- **Interactive demo**: Press 'M' to enable manual control, use arrow keys to balance poles

### Running the CartPole Demo

```bash
# Run the interactive CartPole gym with 6 CartPoles
cargo run --release --features render

# Controls:
# M - Toggle manual control mode
# 1-6 - Select CartPole (when manual control is active)
# Left/Right arrows - Apply force to selected CartPole
# Space - Stop applying force
# R - Reset all CartPoles
# WASD - Move camera
# Mouse - Look around
```

### Testing CartPole Physics

```bash
# Run CartPole-specific tests
cargo test -p physics cartpole

# Run ML environment tests
cargo test -p ml --test 07_stick_balance_env
cargo test -p ml --test 08_cart_pole
cargo test -p ml --test 09_cart_pole_control
cargo test -p ml --test 10_cart_pole_train -- --ignored  # PPO training

## Status

The codebase is early and experimental. Right now it demonstrates the core pieces needed to integrate physics simulation with GPU kernels compiled from WGSL. The intention is to evolve this into a Brax‑like environment where reinforcement learning policies can be trained directly on a differentiable WebGPU simulator.

Contributions and issue reports are welcome as we iterate on the design and expand the feature set.
