@group(0) @binding(0) var<storage, read> cond: array<u32>;
@group(0) @binding(1) var<storage, read> tval: array<f32>;
@group(0) @binding(2) var<storage, read> fval: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = select(fval[i], tval[i], cond[i] != 0u);
}
