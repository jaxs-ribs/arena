// SDF Renderer Shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
}

struct SceneCounts {
    spheres: u32,
    boxes: u32,
    cylinders: u32,
    planes: u32,
}

struct Sphere {
    pos: vec3<f32>,
    _pad: f32,
}

struct Box {
    pos: vec3<f32>,
    _pad1: f32,
    half_extents: vec3<f32>,
    _pad2: f32,
}

struct Cylinder {
    pos: vec3<f32>,
    radius: f32,
    height: f32,
    _pad0: vec3<f32>,
}

struct Plane {
    normal: vec3<f32>,
    d: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> counts: SceneCounts;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> boxes: array<Box>;
@group(0) @binding(4) var<storage, read> cylinders: array<Cylinder>;
@group(0) @binding(5) var<storage, read> planes: array<Plane>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Create a full-screen quad
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// SDF for sphere
fn sdf_sphere(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

// SDF for box
fn sdf_box(p: vec3<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p - center) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// SDF for cylinder - fixed formula
fn sdf_cylinder(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    let offset = p - center;
    let d = vec2<f32>(length(offset.xz), abs(offset.y)) - vec2<f32>(radius, height * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// SDF for plane
fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, d: f32) -> f32 {
    return dot(p, normal) + d;
}

// Scene SDF - combine all objects with proper spacing
fn scene_sdf(p: vec3<f32>) -> f32 {
    var dist = 1000.0;
    
    // Spheres
    for (var i = 0u; i < counts.spheres; i++) {
        let sphere_dist = sdf_sphere(p, spheres[i].pos, 0.5); // Default radius 0.5
        dist = min(dist, sphere_dist);
    }
    
    // Boxes
    for (var i = 0u; i < counts.boxes; i++) {
        let box_dist = sdf_box(p, boxes[i].pos, boxes[i].half_extents);
        dist = min(dist, box_dist);
    }
    
    // Cylinders - add small offset to prevent merging
    for (var i = 0u; i < counts.cylinders; i++) {
        let cylinder_dist = sdf_cylinder(p, cylinders[i].pos, cylinders[i].radius, cylinders[i].height);
        dist = min(dist, cylinder_dist);
    }
    
    // Planes
    for (var i = 0u; i < counts.planes; i++) {
        let plane_dist = sdf_plane(p, planes[i].normal, planes[i].d);
        dist = min(dist, plane_dist);
    }
    
    return dist;
}

// Ray marching with better precision
fn ray_march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    var t = 0.01;  // Start very close to camera
    let max_steps = 128;
    let min_dist = 0.001;
    let max_dist = 100.0;
    
    for (var i = 0; i < max_steps; i++) {
        let p = ray_origin + t * ray_dir;
        let dist = scene_sdf(p);
        
        if (dist < min_dist) {
            return t;
        }
        
        t += dist;  // Full step marching
        
        if (t > max_dist) {
            break;
        }
    }
    
    return -1.0; // No hit
}

// Calculate normal at surface point
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.001;
    return normalize(vec3<f32>(
        scene_sdf(p + vec3<f32>(eps, 0.0, 0.0)) - scene_sdf(p - vec3<f32>(eps, 0.0, 0.0)),
        scene_sdf(p + vec3<f32>(0.0, eps, 0.0)) - scene_sdf(p - vec3<f32>(0.0, eps, 0.0)),
        scene_sdf(p + vec3<f32>(0.0, 0.0, eps)) - scene_sdf(p - vec3<f32>(0.0, 0.0, eps))
    ));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert screen coordinates to NDC
    let screen_size = vec2<f32>(800.0, 600.0);
    let uv = (frag_coord.xy / screen_size) * 2.0 - 1.0;
    
    // Create ray from camera - flip Y to fix upside down
    let ndc = vec4<f32>(uv.x, -uv.y, 0.0, 1.0);
    let world_near = camera.view_proj_inv * ndc;
    let ndc_far = vec4<f32>(uv.x, -uv.y, 1.0, 1.0);
    let world_far = camera.view_proj_inv * ndc_far;
    
    let ray_origin = camera.eye.xyz;
    let ray_dir = normalize((world_far.xyz / world_far.w) - (world_near.xyz / world_near.w));
    
    // Ray march
    let t = ray_march(ray_origin, ray_dir);
    
    if (t < 0.0) {
        // Background
        return vec4<f32>(0.1, 0.2, 0.3, 1.0);
    }
    
    // Hit something
    let hit_pos = ray_origin + t * ray_dir;
    let normal = calc_normal(hit_pos);
    
    // Simple lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let light = max(dot(normal, light_dir), 0.1);
    
    // Color based on object type
    var color = vec3<f32>(0.8, 0.8, 0.8); // Default gray
    
    // Simple shading
    color *= light;
    
    return vec4<f32>(color, 1.0);
}