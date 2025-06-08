@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> _cfg: u32;

@compute @workgroup_size(1)
fn main() {
    var m: f32 = a[0];
    for (var i: u32 = 1u; i < arrayLength(&a); i = i + 1u) {
        m = max(m, a[i]);
    }
    out[0] = m;
}
