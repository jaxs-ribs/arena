// Debug shader to visualize cylinder orientations

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    let x = f32(i32(vertex_index) - 1);
    let y = f32(i32(vertex_index & 1u) * 2 - 1);
    
    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.color = vec3<f32>(0.5, 0.5, 0.5);
    return output;
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
};

struct Cylinder {
    pos: vec3<f32>,
    radius: f32,
    height: f32,
    _pad0: vec3<f32>,
    orientation: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(2) var<storage, read> cylinders: array<Cylinder>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.position.xy / camera.resolution;
    
    // Debug visualization: show cylinder orientations as colors
    if (uv.y < 0.1) {
        // Bottom strip: show raw orientation data
        let num_cylinders = arrayLength(&cylinders);
        if (num_cylinders > 0u) {
            let idx = min(u32(uv.x * f32(num_cylinders)), num_cylinders - 1u);
            let cyl = cylinders[idx];
            
            // Map quaternion to color
            // X,Y,Z components map to R,G,B
            // W component affects brightness
            let color = vec3<f32>(
                cyl.orientation.x * 0.5 + 0.5,
                cyl.orientation.y * 0.5 + 0.5,
                cyl.orientation.z * 0.5 + 0.5
            ) * cyl.orientation.w;
            
            return vec4<f32>(color, 1.0);
        }
    } else if (uv.y < 0.2) {
        // Second strip: show cylinder positions
        let num_cylinders = arrayLength(&cylinders);
        if (num_cylinders > 0u) {
            let idx = min(u32(uv.x * f32(num_cylinders)), num_cylinders - 1u);
            let cyl = cylinders[idx];
            
            // Map position to color
            let color = vec3<f32>(
                cyl.pos.x * 0.1 + 0.5,
                cyl.pos.y * 0.1 + 0.5,
                cyl.pos.z * 0.1 + 0.5
            );
            
            return vec4<f32>(color, 1.0);
        }
    }
    
    // Rest of screen: gradient based on orientation magnitude
    if (arrayLength(&cylinders) > 0u) {
        let cyl = cylinders[0];
        let angle = acos(clamp(cyl.orientation.w, -1.0, 1.0)) * 2.0;
        let normalized_angle = angle / 3.14159;
        
        return vec4<f32>(normalized_angle, 1.0 - normalized_angle, 0.0, 1.0);
    }
    
    return vec4<f32>(0.1, 0.1, 0.1, 1.0);
}