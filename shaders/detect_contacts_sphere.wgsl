struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Body {
    pos : Vec3,
};

struct Contact {
    body_index : u32,
    normal : Vec3,
    depth : f32,
};

@group(0) @binding(0) var<storage, read> bodies : array<Body>;
@group(0) @binding(1) var<storage, read_write> contacts : array<Contact>;

@compute @workgroup_size(1)
fn main() {
    let count = arrayLength(&bodies);
    var out_idx : u32 = 0u;
    for (var i : u32 = 0u; i < count; i = i + 1u) {
        let a = bodies[i];
        for (var j : u32 = i + 1u; j < count; j = j + 1u) {
            let b = bodies[j];
            let dx = b.pos.x - a.pos.x;
            let dy = b.pos.y - a.pos.y;
            let dz = b.pos.z - a.pos.z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rad_sum = 2.0;
            if (dist_sq < rad_sum * rad_sum) {
                let dist = sqrt(dist_sq);
                var n = vec3<f32>(1.0, 0.0, 0.0);
                if (dist > 0.0) {
                    n = vec3<f32>(dx / dist, dy / dist, dz / dist);
                }
                let depth = rad_sum - dist;
                if (out_idx < arrayLength(&contacts)) {
                    var c : Contact;
                    c.body_index = i;
                    c.normal = Vec3(-n.x, -n.y, -n.z);
                    c.depth = depth * 0.5;
                    contacts[out_idx] = c;
                }
                if (out_idx + 1u < arrayLength(&contacts)) {
                    var c2 : Contact;
                    c2.body_index = j;
                    c2.normal = Vec3(n.x, n.y, n.z);
                    c2.depth = depth * 0.5;
                    contacts[out_idx + 1u] = c2;
                }
                out_idx = out_idx + 2u;
            }
        }
    }
}
