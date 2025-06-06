struct Sphere {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct Plane {
    normal: vec3<f32>,
    height: f32,
};

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    resolution: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> plane: Plane;

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    var out: VsOut;
    out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    return out;
}

fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
    return dot(p, n) + h;
}

fn sdf_scene(p: vec3<f32>) -> f32 {
    var d = 1e10;
    for (var i = 0u; i < arrayLength(&spheres); i = i + 1u) {
        d = min(d, sdf_sphere(p - spheres[i].pos.xyz, 0.1));
    }
    d = min(d, sdf_plane(p, plane.normal, plane.height));
    return d;
}

fn get_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * 0.5773;
    let eps = 0.0005;
    return normalize(
        e.xyy * sdf_scene(p + e.xyy * eps) +
        e.yyx * sdf_scene(p + e.yyx * eps) +
        e.yxy * sdf_scene(p + e.yxy * eps) +
        e.xxx * sdf_scene(p + e.xxx * eps)
    );
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.position.xy / camera.resolution.xy;
    let ro = camera.position;

    let ndc = vec3<f32>(uv * 2.0 - 1.0, 1.0);
    let world = camera.inv_view_proj * vec4<f32>(ndc, 1.0);
    let rd = normalize(world.xyz / world.w - ro);

    var t = 0.0;
    for (var i = 0; i < 100; i = i + 1) {
        let p = ro + rd * t;
        let d = sdf_scene(p);
        if (d < 0.001) {
            let n = get_normal(p);
            let light_pos = vec3<f32>(5.0, 5.0, 5.0);
            let light_dir = normalize(light_pos - p);
            let diffuse = max(dot(n, light_dir), 0.0);
            let object_color = vec3(0.8); // Gray
            let plane_color = vec3(0.5, 0.5, 0.5); // Darker Gray for plane
            
            // Check if we hit the plane to color it differently
            let plane_dist = sdf_plane(p, plane.normal, plane.height);
            var color = object_color;
            if (plane_dist < 0.002) { // if we are very close to plane surface
                color = plane_color;
            }

            return vec4<f32>(color * diffuse, 1.0);
        }
        t = t + d;
        if (t > 100.0) {
            break;
        }
    }

    return vec4<f32>(0.1, 0.2, 0.3, 1.0); // Background color
}
