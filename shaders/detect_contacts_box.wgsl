struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Body {
    pos : Vec3,
    _pad1: f32,
    vel: Vec3,
    _pad2: f32,
    orientation: vec4<f32>,
    angular_vel: Vec3,
    _pad3: f32,
};

struct Contact {
    body_index : u32,
    normal : Vec3,
    depth : f32,
};

struct BoxShape {
    center: vec3<f32>,
    _pad1: f32,
    half_extents: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> box_data : BoxShape;
@group(0) @binding(2) var<storage, read_write> contacts : array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    let center = box_data.center;
    let half_extents = box_data.half_extents;

    let min_x = center.x - half_extents.x;
    let max_x = center.x + half_extents.x;
    let min_y = center.y - half_extents.y;
    let max_y = center.y + half_extents.y;
    let min_z = center.z - half_extents.z;
    let max_z = center.z + half_extents.z;

    atomicStore(&contacts[0], 0u);

    for (var i : u32 = 0u; i < arrayLength(&bodies); i = i + 1u) {
        let p = bodies[i].pos;
        let clamped_x = clamp(p.x, min_x, max_x);
        let clamped_y = clamp(p.y, min_y, max_y);
        let clamped_z = clamp(p.z, min_z, max_z);

        let dx = p.x - clamped_x;
        let dy = p.y - clamped_y;
        let dz = p.z - clamped_z;
        let dist_sq = dx * dx + dy * dy + dz * dz;

        var emit = false;
        var n = vec3<f32>(0.0, 0.0, 0.0);
        var depth = 0.0;

        if (dist_sq > 0.0) {
            if (dist_sq < 1.0) {
                let dist = sqrt(dist_sq);
                n = vec3<f32>(dx / dist, dy / dist, dz / dist);
                depth = 1.0 - dist;
                emit = true;
            }
        } else if p.x >= min_x && p.x <= max_x &&
                  p.y >= min_y && p.y <= max_y &&
                  p.z >= min_z && p.z <= max_z {
            var min_d = max_x - p.x;
            n = vec3<f32>(1.0, 0.0, 0.0);
            if (p.x - min_x < min_d) { min_d = p.x - min_x; n = vec3<f32>(-1.0,0.0,0.0); }
            if (max_y - p.y < min_d) { min_d = max_y - p.y; n = vec3<f32>(0.0,1.0,0.0); }
            if (p.y - min_y < min_d) { min_d = p.y - min_y; n = vec3<f32>(0.0,-1.0,0.0); }
            if (max_z - p.z < min_d) { min_d = max_z - p.z; n = vec3<f32>(0.0,0.0,1.0); }
            if (p.z - min_z < min_d) { min_d = p.z - min_z; n = vec3<f32>(0.0,0.0,-1.0); }
            depth = 1.0 + min_d;
            emit = true;
        }

        if (emit) {
            let out_idx = atomicAdd(&contacts[0], 1u);
            if (out_idx < arrayLength(&contacts) - 1u) {
            var c : Contact;
            c.body_index = i;
            c.normal = Vec3(n.x, n.y, n.z);
            c.depth = depth;
                let contact_as_u32 = bitcast<vec4<u32>>(c);
                atomicStore(&contacts[4 * out_idx + 4], contact_as_u32.x);
                atomicStore(&contacts[4 * out_idx + 5], contact_as_u32.y);
                atomicStore(&contacts[4 * out_idx + 6], contact_as_u32.z);
                atomicStore(&contacts[4 * out_idx + 7], contact_as_u32.w);
            }
        }
    }
}
