# JAXS Renderer

The JAXS renderer is a **ray-marching SDF (Signed Distance Field) renderer** that visualizes physics simulations in real-time. It provides a clean, efficient way to render primitive shapes without polygon meshes.

## Overall Architecture

The renderer uses a modular architecture with clear separation of concerns:

```
Physics Simulation → GPU Types → Scene Manager → GPU Buffers → WGSL Shader → Screen
                                                       ↑
                                                    Camera
```

## Key Concepts: Ray Marching & SDFs

Unlike traditional polygon rendering, this renderer uses **ray marching through signed distance fields**:

- **SDF (Signed Distance Field)**: A function that returns the distance from any point in 3D space to the nearest surface. Negative values mean you're inside the object.
- **Ray Marching**: Instead of testing ray-triangle intersections, we "march" along a ray by the SDF distance at each step until we hit a surface.

## Module Breakdown

### camera.rs - First-Person Camera System
- **Camera**: Stores position, orientation (yaw/pitch), and projection parameters
- **CameraController**: Handles WASD movement and mouse look
- Generates view-projection matrices for transforming between world and screen space

### gpu_types.rs - CPU↔GPU Data Bridge
- Defines GPU-compatible structs (all are `#[repr(C)]` and implement `Pod`)
- Each physics primitive has a GPU counterpart:
  - `Sphere` → `SphereGpu` (includes material properties like friction/restitution)
  - `BoxBody` → `BoxGpu`
  - `Cylinder` → `CylinderGpu`
  - `Plane` → `PlaneGpu`
- `CameraUniform`: Contains view matrices and eye position for the shader

### scene.rs - Scene Management
- `SceneManager`: Updates GPU buffers when physics simulation advances
- Converts physics objects to GPU format and uploads them
- Maintains a count buffer so the shader knows how many objects to render

### pipeline.rs - GPU Pipeline Setup
- Creates the render pipeline with vertex/fragment shader stages
- Defines bind group layout (how data is passed to shaders):
  - Binding 0: Camera uniform
  - Binding 1: Object counts
  - Bindings 2-5: Storage buffers for each primitive type
- Creates a fullscreen quad (4 vertices) that the fragment shader will ray march through

### renderer.rs - Main Orchestrator
- Manages the window, surface, and GPU device
- Handles events (resize, keyboard, mouse)
- Render loop:
  1. Update camera based on input
  2. Update scene buffers with physics data
  3. Render frame by drawing the fullscreen quad

## The WGSL Shader - Where the Magic Happens

The shader (`renderer.wgsl`) implements the actual ray marching algorithm:

### Vertex Shader (`vs_main`)
- Simply outputs a fullscreen quad (-1 to 1 in NDC)
- The fragment shader will be called for every pixel

### Fragment Shader (`fs_main`)
1. **Ray Generation**: For each pixel, creates a ray from the camera through that pixel
2. **Ray Marching**: Steps along the ray using the scene SDF
3. **Hit Detection**: When close enough to a surface, records what was hit
4. **Shading**: Calculates lighting and applies material-based colors

### SDF Functions
Each primitive has its own SDF function:
- `sdf_sphere`: `distance = length(p - center) - radius`
- `sdf_box`: Uses the "rounded box" SDF formula
- `sdf_cylinder`: Combines 2D circle distance (XZ) with height bounds
- `sdf_plane`: Dot product with normal, constrained to finite extents

### Scene SDF (`scene_sdf_detailed`)
- Loops through all objects and finds the closest one
- Returns both distance and which object was hit
- This is called ~100+ times per pixel during ray marching!

### Ray Marching Algorithm
```wgsl
for each step:
    p = ray_origin + t * ray_direction
    distance = scene_sdf(p)
    if distance < threshold:
        // Hit! Do binary refinement for accuracy
        return hit info
    t += distance  // Safe to step by SDF distance
```

### Shading
- Calculates surface normal using finite differences of the SDF
- Simple directional lighting (dot product with light direction)
- Colors based on object type and material properties:
  - **Spheres**: Color varies with friction (red) and restitution (green)
  - **Boxes**: Brown
  - **Cylinders**: Green
  - **Planes**: Light blue-gray

## Performance Considerations

- **Adaptive convergence threshold**: Scales with distance to maintain precision
- **Binary refinement**: When a hit is detected, backs up and refines for accuracy
- **Storage buffers**: All objects are stored in GPU buffers, no CPU↔GPU transfer during rendering
- **Parallelism**: Each pixel is computed independently on the GPU

## Why Ray Marching for Physics Visualization?

1. **Perfect spheres**: No tessellation artifacts for spheres
2. **Easy to add primitives**: Just need an SDF function
3. **Smooth blending**: Can use smooth minimum for soft object interactions
4. **No mesh generation**: Physics objects render directly without triangulation
5. **Great for debugging**: Can easily visualize implicit surfaces, distance fields, etc.

## Usage

```rust
// Create renderer with default config
let renderer_config = RendererConfig::default();
let (mut renderer, event_loop) = Renderer::new(renderer_config)?;

// In your render loop:
renderer.update_scene(&spheres, &boxes, &cylinders, &planes);
renderer.render(dt)?;
```

## Controls

- **WASD**: Move camera
- **Mouse**: Look around
- **Space**: Move up
- **Shift**: Move down
- **F**: Toggle fullscreen
- **Escape**: Release mouse capture

## Running the Demo

Run with the `render` feature enabled:

```bash
cargo run --features render
```

The renderer provides a clean, real-time visualization of the physics simulation with material properties visible through color coding, making it ideal for debugging and understanding simulation behavior.