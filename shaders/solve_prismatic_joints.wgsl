struct Body { pos : vec3<f32>; };
struct Joint { body_a: u32; body_b: u32; anchor_a: vec3<f32>; anchor_b: vec3<f32>; axis: vec3<f32>; lower_limit: f32; upper_limit: f32; motor_speed: f32; motor_max_force: f32; enable_motor: u32; enable_limit: u32; _pad: f32; };
struct Params { compliance: f32; _pad: vec3<f32>; };
@group(0) @binding(0) var<storage, read_write> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> joints : array<Joint>;
@group(0) @binding(2) var<uniform> _params : Params;
@compute @workgroup_size(1)
fn main() { }
