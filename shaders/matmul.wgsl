@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec3<u32>; // M, N, K

@compute @workgroup_size(1)
fn main() {
    let M = cfg.x;
    let N = cfg.y;
    let K = cfg.z;
    for (var m: u32 = 0u; m < M; m = m + 1u) {
        for (var n: u32 = 0u; n < N; n = n + 1u) {
            var sum: f32 = 0.0;
            for (var k: u32 = 0u; k < K; k = k + 1u) {
                let a_idx = m * K + k;
                let b_idx = k * N + n;
                sum = sum + a[a_idx] * b[b_idx];
            }
            out[m * N + n] = sum;
        }
    }
}
