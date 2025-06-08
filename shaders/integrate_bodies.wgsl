struct Sphere {
  pos : vec3<f32>,
  _pad1: f32,
  vel : vec3<f32>,
  _pad2: f32,
  orientation : vec4<f32>,
  angular_vel : vec3<f32>,
  _pad3: f32,
};

struct RawPhysParams {
  gravity: vec3<f32>,
  dt: f32,
}

@group(0) @binding(0) var<storage, read_write> spheres : array<Sphere>;
@group(0) @binding(1) var<uniform>            params  : RawPhysParams;
@group(0) @binding(2) var<storage, read>      forces  : array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&spheres)) { return; }

  var s = spheres[idx];
  let dt = params.dt;
  let g = params.gravity;
  let f = vec3<f32>(forces[idx], 0.0);
  let acc = g + f;
  s.vel += acc * dt;
  s.pos += s.vel * dt;

  let half_dt = 0.5 * dt;
  let ox = s.angular_vel.x * half_dt;
  let oy = s.angular_vel.y * half_dt;
  let oz = s.angular_vel.z * half_dt;
  let qx = s.orientation.x;
  let qy = s.orientation.y;
  let qz = s.orientation.z;
  let qw = s.orientation.w;
  s.orientation.x += ox * qw + oy * qz - oz * qy;
  s.orientation.y += oy * qw + oz * qx - ox * qz;
  s.orientation.z += oz * qw + ox * qy - oy * qx;
  s.orientation.w += -ox * qx - oy * qy - oz * qz;
  
  // floor at y=0
  if (s.pos.y < 0.0) {
      s.pos.y = 0.0;
      s.vel.y = 0.0; // Stop vertical velocity on impact
  }
  spheres[idx] = s;
} 
