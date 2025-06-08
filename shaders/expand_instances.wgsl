@group(0) @binding(0) var<storage, read> tmpl: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> cfg: u32; // count

@compute @workgroup_size(1)
fn main() {
    let count = cfg;
    let len = arrayLength(&tmpl);
    for (var c: u32 = 0u; c < count; c = c + 1u) {
        for (var i: u32 = 0u; i < len; i = i + 1u) {
            out[c * len + i] = tmpl[i];
        }
    }
}
