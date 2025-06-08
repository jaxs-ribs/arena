# Compute Crate Backend Architecture

This document outlines the architecture of the `compute` crate, which is designed to support multiple computation backends for physics simulations and other operations.

## Overview

The core of this crate is the `Backend` trait, located in `src/backend.rs`. This trait defines a common interface for all computational kernels. By programming against this trait, the physics engine and other high-level components can remain agnostic about the underlying hardware (CPU or GPU).

We currently have two primary backend implementations:

-   `CpuBackend`: Utilizes the standard CPU for all computations. The implementations are located in `src/cpu.rs` and wrap the existing kernel logic found in `src/kernels/`.
-   `GpuBackend`: A placeholder for a future `wgpu`-based backend. Its methods are currently unimplemented. The implementation is in `src/gpu.rs`.

## Backend Selection

The active backend is selected at compile time using feature flags:

-   `--features cpu`: Compiles and uses the `CpuBackend`.
-   `--features gpu`: Compiles and uses the `GpuBackend`.

If no feature is specified, the build may fail or result in a no-op backend. The default is managed in the workspace `Cargo.toml`.

## How to Add a New GPU Kernel

When you are ready to implement a new GPU-accelerated kernel, follow these steps:

1.  **Add to the `Backend` Trait**: If the operation is new, first add its method signature to the `Backend` trait in `src/backend.rs`.

2.  **Implement the Kernel in `GpuBackend`**: Open `src/gpu.rs` and add the implementation for the new kernel within the `impl Backend for GpuBackend` block. This will likely involve:
    -   Writing a WGSL shader for the operation.
    -   Creating a `wgpu` compute pipeline.
    -   Dispatching the compute shader with the appropriate buffers and bind groups.

3.  **Follow Existing Patterns**: Look at the existing CPU implementations in `src/cpu.rs` and the kernel definitions in `src/kernels/` to understand the expected inputs and outputs for your operation. Your GPU implementation should match this behavior.

4.  **Test**: Ensure you run tests with the `--features gpu` flag to validate your new kernel. 