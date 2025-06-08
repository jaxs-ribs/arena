@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> _cfg: u32;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&out);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        out[i] = f32(i) * 0.1;
    }
}
