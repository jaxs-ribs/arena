@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;
@group(0) @binding(3) var<uniform> _cfg: u32;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&values);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let idx = indices[i];
        acc[idx] = acc[idx] + values[i];
    }
}
