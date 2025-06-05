@group(0) @binding(0) var<storage, read>  a  : array<f32>;
@group(0) @binding(1) var<storage, read>  b  : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>  config : u32;   // 0=add,1=mul,2=where

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }

  let op = config;
  let av = a[i];
  let bv = b[i];

  var r = 0.0;
  if      (op == 0u) { r = av + bv; }
  else if (op == 1u) { r = av * bv; }
  else {
      r = select(bv, av, bv == 0.0);
  }

  out[i] = r;
}
