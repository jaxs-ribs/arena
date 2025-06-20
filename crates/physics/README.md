# Physics Crate

A differentiable physics simulation engine designed for reinforcement learning environments.

## Features

### Rigid Bodies
- **Spheres**: Point masses with radius
- **Boxes**: Rectangular bodies with half-extents
- **Cylinders**: Cylindrical bodies with radius and height
- **Planes**: Static infinite planes for ground/walls

### Collision Detection & Response
- Sphere-sphere, sphere-plane, sphere-box, sphere-cylinder collisions
- Box-plane and cylinder-plane collisions
- Material properties: friction and restitution coefficients
- Position-based collision resolution with impulse-based dynamics

### Constraints
- **Distance Joints**: Maintain fixed distance between bodies
- **Revolute Joints**: Hinge constraints allowing rotation around an axis
- **2D Motion Constraints**: Lock bodies to 2D plane for specific simulations

### Environments
- **CartPole**: Classic control task with configurable parameters
  - Grid layout for parallel training
  - Automatic failure detection and reset
  - State observation and action application

## Usage

```rust
use physics::{PhysicsSim, CartPole, CartPoleConfig, Vec3};

// Create simulation
let mut sim = PhysicsSim::new();
sim.params.gravity = Vec3::new(0.0, -9.81, 0.0);

// Add ground plane
sim.add_plane(Vec3::new(0.0, 1.0, 0.0), 0.0, Vec2::new(10.0, 10.0));

// Create CartPole
let config = CartPoleConfig::default();
let cartpole = CartPole::new(&mut sim, Vec3::ZERO, config);

// Run simulation
for _ in 0..100 {
    cartpole.apply_force(&mut sim, 0.5); // Apply rightward force
    sim.step_cpu();
    
    // Check failure
    if cartpole.check_failure(&sim) {
        cartpole.reset(&mut sim);
    }
}
```

## Architecture

The physics engine uses:
- Semi-implicit Euler integration
- Position-based dynamics for constraint solving
- Spatial hashing for broad-phase collision detection
- GPU compute kernels (via `compute` crate) for parallel processing

## Testing

```bash
# Run all physics tests
cargo test -p physics

# Run specific test suites
cargo test -p physics collision     # Collision tests
cargo test -p physics revolute      # Revolute joint tests
cargo test -p physics cartpole      # CartPole environment tests
```

## Performance

The engine is designed for:
- Real-time simulation (60Hz+)
- Parallel environments for RL training
- GPU acceleration for large-scale simulations
- Differentiable operations for gradient-based optimization