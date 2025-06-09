struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Body {
    pos : Vec3,
};

struct Joint {
    body_a : u32,
    body_b : u32,
    rest_length : f32,
    _pad : u32,
};

struct SolveParams {
    compliance : f32,
    _pad : vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> joints : array<Joint>;
@group(0) @binding(2) var<uniform> _params : SolveParams;

@compute @workgroup_size(1)
fn main() {
    let nj = arrayLength(&joints);
    for (var i : u32 = 0u; i < nj; i = i + 1u) {
        let jnt = joints[i];
        if (jnt.body_a >= arrayLength(&bodies) || jnt.body_b >= arrayLength(&bodies)) {
            continue;
        }
        var pa = bodies[jnt.body_a].pos;
        var pb = bodies[jnt.body_b].pos;

        let dx = vec3<f32>(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        let len_sq = dot(dx, dx);
        if (len_sq == 0.0) { continue; }
        let len = sqrt(len_sq);
        let diff = (len - jnt.rest_length) / len * 0.5;
        pa.x += dx.x * diff;
        pa.y += dx.y * diff;
        pa.z += dx.z * diff;
        pb.x -= dx.x * diff;
        pb.y -= dx.y * diff;
        pb.z -= dx.z * diff;
        bodies[jnt.body_a].pos = pa;
        bodies[jnt.body_b].pos = pb;
    }
}
