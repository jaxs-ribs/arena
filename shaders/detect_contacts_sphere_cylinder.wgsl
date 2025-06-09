struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
};

struct Contact {
    sphere_index: u32,
    cylinder_index: u32,
    contact_point: Vec3,
    normal: Vec3,
    depth: f32,
    _pad: Vec3,
};

@group(0) @binding(0) var<storage, read> _spheres: array<Vec3>;
@group(0) @binding(1) var<storage, read> _cylinders: array<Vec3>;
@group(0) @binding(2) var<storage, read_write> _contacts: array<Contact>;

@compute @workgroup_size(1)
fn main() {
    // placeholder implementation
}
