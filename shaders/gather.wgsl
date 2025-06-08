@group(0) @binding(0) var<storage, read> data_in: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> _cfg: u32;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&indices);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        out[i] = data_in[indices[i]];
    }
}
