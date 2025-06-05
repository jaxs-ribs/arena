struct VsOut {
    @builtin(position) position: vec4<f32>;
    @builtin(point_size) size: f32;
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.size = 8.0;
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.8, 1.0, 1.0);
}
