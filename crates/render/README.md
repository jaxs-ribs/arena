# Render Crate

This crate contains a small WGPU-based renderer used by the runtime binary.
It can visualize the spheres produced by the physics simulator.

### Running the demo

Run the runtime with the `render` feature and pass `--draw` to enable
rendering:

```bash
cargo run -p runtime --features render -- --draw
```

A window will appear showing the positions of simulated spheres.
