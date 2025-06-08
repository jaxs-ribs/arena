@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> config: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let a_shape_dim1 = config[0];
    let b_len = config[1];

    for (var i = 0u; i < a_shape_dim1; i = i + 1u) {
        let a_idx = batch_idx * a_shape_dim1 + i;
        let b_idx = i % b_len;
        out[a_idx] = a[a_idx] + b[b_idx];
    }
} 