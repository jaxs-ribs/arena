@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main() {
    for (var i: u32 = 0u; i < arrayLength(&a); i = i + 1u) {
        out[i] = a[i] + b[i];
    }
}
