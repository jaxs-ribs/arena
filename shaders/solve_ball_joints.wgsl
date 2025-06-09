struct Body { pos : vec3<f32>; };
struct Joint { body_a: u32; body_b: u32; anchor_a: vec3<f32>; anchor_b: vec3<f32>; _pad: vec2<f32>; };
struct Params { compliance: f32; _pad: vec3<f32>; };
@group(0) @binding(0) var<storage, read_write> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> joints : array<Joint>;
@group(0) @binding(2) var<uniform> _params : Params;
@compute @workgroup_size(1)
fn main() { }
