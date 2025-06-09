struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Body {
    pos : Vec3,
};

struct Contact {
    index : u32,
    penetration : f32,
};

@group(0) @binding(0) var<storage, read> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> plane : array<f32>;
@group(0) @binding(2) var<storage, read_write> contacts : array<Contact>;

@compute @workgroup_size(1)
fn main() {
    let height = plane[0];
    var out_idx : u32 = 0u;
    for (var i : u32 = 0u; i < arrayLength(&bodies); i = i + 1u) {
        let p = bodies[i].pos;
        if (p.y < height && out_idx < arrayLength(&contacts)) {
            var c : Contact;
            c.index = i;
            c.penetration = height - p.y;
            contacts[out_idx] = c;
            out_idx = out_idx + 1u;
        }
    }
}
