@group(0) @binding(0) var<storage, read> data_in: array<f32>;
@group(0) @binding(1) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> _cfg: u32;

@compute @workgroup_size(1)
fn main() {
    let seg_count = arrayLength(&segments);
    for (var s: u32 = 0u; s < seg_count; s = s + 1u) {
        let start = segments[s];
        let end = if (s + 1u < seg_count) { segments[s + 1u] } else { arrayLength(&data_in) };
        var sum: f32 = 0.0;
        for (var i: u32 = start; i < end; i = i + 1u) {
            sum = sum + data_in[i];
        }
        out[s] = sum;
    }
}
