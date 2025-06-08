struct Sphere {
  pos : vec3<f32>,
  vel : vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> spheres : array<Sphere>;
@group(0) @binding(1) var<uniform>            params  : vec4<f32>; // xyz: gravity, w: dt
@group(0) @binding(2) var<storage, read>      forces  : array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&spheres)) { return; }

  var s = spheres[idx];
  let dt = params.w;
  let g = params.xyz;
  let f = vec3<f32>(forces[idx], 0.0);
  let acc = g + f;
  s.pos += s.vel * dt + 0.5 * acc * dt * dt;
  s.vel += acc * dt;
  
  // floor at y=0
  if (s.pos.y < 0.0) {
      s.pos.y = 0.0;
      s.vel.y = 0.0; // Stop vertical velocity on impact
  }
  spheres[idx] = s;
} 