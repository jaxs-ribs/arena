struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Sphere {
    pos : Vec3,
    _pad1: f32,
    vel : Vec3,
    _pad2: f32,
    orientation : vec4<f32>,
    angular_vel : Vec3,
    _pad3: f32,
};

struct Contact {
    body_index : u32,
    normal : Vec3,
    depth : f32,
};

@group(0) @binding(0) var<storage, read_write> bodies : array<Sphere>;
@group(0) @binding(1) var<storage, read> contacts : array<Contact>;
@group(0) @binding(2) var<uniform> _params : u32;

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&contacts);
    for (var i : u32 = 0u; i < n; i = i + 1u) {
        let c = contacts[i];
        if (c.body_index >= arrayLength(&bodies)) { continue; }
        var b = bodies[c.body_index];
        b.pos.x += c.normal.x * c.depth;
        b.pos.y += c.normal.y * c.depth;
        b.pos.z += c.normal.z * c.depth;
        bodies[c.body_index] = b;
    }
}
