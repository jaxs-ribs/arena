struct Sphere {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let sphere = spheres[instance_index];

    // Generate a quad
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(1.0, -1.0); }
        case 2u: { pos = vec2<f32>(1.0, 1.0); }
        case 3u: { pos = vec2<f32>(-1.0, -1.0); }
        case 4u: { pos = vec2<f32>(1.0, 1.0); }
        case 5u: { pos = vec2<f32>(-1.0, 1.0); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    let radius = 0.1;

    var out: VsOut;
    out.position = vec4<f32>(sphere.pos.xy + pos * radius, 0.0, 1.0);
    out.local_pos = pos;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dist = length(in.local_pos) - 1.0;
    let smooth_dist = 1.0 - smoothstep(-0.02, 0.02, dist);
    return vec4<f32>(smooth_dist, smooth_dist, smooth_dist, 1.0);
}
