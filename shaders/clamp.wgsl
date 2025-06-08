@group(0) @binding(0) var<storage, read> val: array<f32>;
@group(0) @binding(1) var<storage, read> min_v: array<f32>;
@group(0) @binding(2) var<storage, read> max_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> _config: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = clamp(val[i], min_v[i], max_v[i]);
}
