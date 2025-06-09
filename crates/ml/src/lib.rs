//! Machine learning utilities built on the differentiable physics primitives.
//!
//! This crate provides tensor operations, basic neural network layers and a few
//! reinforcement learning helpers. It is **not** a full ML framework but rather
//! contains only the pieces needed for the included examples and tests.

pub mod graph;
pub mod nn;
pub mod optim;
pub mod recorder;
pub mod env;
pub mod rl;
pub mod stick_balance;
pub mod tape;
pub mod tensor;

pub use tensor::Tensor;
pub use stick_balance::StickBalanceEnv;
pub use env::Env;
