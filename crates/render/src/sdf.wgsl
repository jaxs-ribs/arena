struct Camera {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
};

struct SceneCounts {
    spheres: u32,
    boxes: u32,
    cylinders: u32,
    planes: u32,
};

struct SphereGpu {
    pos: vec3<f32>,
    _pad: f32,
};

struct BoxGpu {
    pos: vec3<f32>,
    _pad1: f32,
    half_extents: vec3<f32>,
    _pad2: f32,
};

struct CylinderGpu {
    pos: vec3<f32>,
    radius: f32,
    height: f32,
    _pad0: vec3<f32>,
};

struct PlaneGpu {
    normal: vec3<f32>,
    d: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> counts: SceneCounts;
@group(0) @binding(2) var<storage> spheres: array<SphereGpu>;
@group(0) @binding(3) var<storage> boxes: array<BoxGpu>;
@group(0) @binding(4) var<storage> cylinders: array<CylinderGpu>;
@group(0) @binding(5) var<storage> planes: array<PlaneGpu>;

fn sdf_sphere(p: vec3<f32>, c: vec3<f32>) -> f32 {
    return length(p - c) - 1.0;
}

fn sdf_box(p: vec3<f32>, c: vec3<f32>, h: vec3<f32>) -> f32 {
    let q = abs(p - c) - h;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_cylinder(p: vec3<f32>, c: vec3<f32>, r: f32, h: f32) -> f32 {
    let d = p - c;
    let q = vec2<f32>(length(d.xz) - r, abs(d.y) - h * 0.5);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
}

fn sdf_plane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32 {
    return dot(p, n) + d;
}

fn scene_sdf(p: vec3<f32>) -> f32 {
    var dist = 1e9;
    var i: u32 = 0u;
    loop {
        if i >= counts.spheres { break; }
        dist = min(dist, sdf_sphere(p, spheres[i].pos));
        i = i + 1u;
    }
    i = 0u;
    loop {
        if i >= counts.boxes { break; }
        dist = min(dist, sdf_box(p, boxes[i].pos, boxes[i].half_extents));
        i = i + 1u;
    }
    i = 0u;
    loop {
        if i >= counts.cylinders { break; }
        dist = min(dist, sdf_cylinder(p, cylinders[i].pos, cylinders[i].radius, cylinders[i].height));
        i = i + 1u;
    }
    i = 0u;
    loop {
        if i >= counts.planes { break; }
        dist = min(dist, sdf_plane(p, planes[i].normal, planes[i].d));
        i = i + 1u;
    }
    return dist;
}

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    var out: VertexOut;
    out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = pos[idx];
    return out;
}

fn get_ray_dir(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv, 1.0, 1.0);
    let world = camera.view_proj_inv * ndc;
    return normalize(world.xyz / world.w - camera.eye.xyz);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var ro = camera.eye.xyz;
    var rd = get_ray_dir(in.uv);
    var t = 0.0;
    var hit = false;
    for(var i = 0u; i < 100u; i = i + 1u) {
        let p = ro + rd * t;
        let d = scene_sdf(p);
        if d < 0.001 {
            hit = true;
            break;
        }
        t = t + d;
        if t > 100.0 { break; }
    }

    if !hit {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let p = ro + rd * t;
    let e = 0.0005;
    let n = normalize(vec3<f32>(
        scene_sdf(p + vec3<f32>(e,0.0,0.0)) - scene_sdf(p - vec3<f32>(e,0.0,0.0)),
        scene_sdf(p + vec3<f32>(0.0,e,0.0)) - scene_sdf(p - vec3<f32>(0.0,e,0.0)),
        scene_sdf(p + vec3<f32>(0.0,0.0,e)) - scene_sdf(p - vec3<f32>(0.0,0.0,e))
    ));
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diff = clamp(dot(n, light_dir), 0.0, 1.0);
    return vec4<f32>(diff, diff, diff, 1.0);
}
