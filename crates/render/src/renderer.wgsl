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
    radius: f32,
    friction: f32,
    restitution: f32,
    _pad: vec2<f32>,
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
    extents: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> counts: SceneCounts;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> boxes: array<Box>;
@group(0) @binding(4) var<storage, read> cylinders: array<Cylinder>;
@group(0) @binding(5) var<storage, read> planes: array<Plane>;

// Define extents for the ground plane
const plane_extents: vec2<f32> = vec2<f32>(25.0, 25.0);

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

// SDF for plane, rendered as a thin box
fn sdf_plane(p: vec3<f32>, pl: Plane) -> f32 {
    // If extents are zero, treat as an infinite plane
    if (pl.extents.x <= 0.0 || pl.extents.y <= 0.0) {
        return dot(p, pl.normal) + pl.d;
    }

    // Create a thin box for the plane
    let thickness = 0.1; // 10cm thick plane
    
    // Transform point to plane's local space
    // For a y-up plane at y=0, this is simple
    let local_p = vec3<f32>(p.x, p.y - pl.d, p.z);
    
    // Box half extents: width, thickness/2, depth
    let half_extents = vec3<f32>(pl.extents.x, thickness * 0.5, pl.extents.y);
    
    // Standard box SDF
    let q = abs(local_p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Smooth minimum function to reduce morphing artifacts
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Structure to return both distance and object information
struct SceneResult {
    distance: f32,
    object_type: u32, // 0=sphere, 1=box, 2=cylinder, 3=plane
    object_index: u32,
}

// Scene SDF - combine all objects and track which was closest
fn scene_sdf_detailed(p: vec3<f32>) -> SceneResult {
    var result = SceneResult(1000.0, 0u, 0u);
    
    // Spheres
    for (var i = 0u; i < counts.spheres; i++) {
        let sphere_dist = sdf_sphere(p, spheres[i].pos, spheres[i].radius);
        if (sphere_dist < result.distance) {
            result.distance = sphere_dist;
            result.object_type = 0u;
            result.object_index = i;
        }
    }
    
    // Boxes
    for (var i = 0u; i < counts.boxes; i++) {
        let box_dist = sdf_box(p, boxes[i].pos, boxes[i].half_extents);
        if (box_dist < result.distance) {
            result.distance = box_dist;
            result.object_type = 1u;
            result.object_index = i;
        }
    }
    
    // Cylinders
    for (var i = 0u; i < counts.cylinders; i++) {
        let cylinder_dist = sdf_cylinder(p, cylinders[i].pos, cylinders[i].radius, cylinders[i].height);
        if (cylinder_dist < result.distance) {
            result.distance = cylinder_dist;
            result.object_type = 2u;
            result.object_index = i;
        }
    }
    
    // Planes
    for (var i = 0u; i < counts.planes; i++) {
        let pl = planes[i];
        let plane_dist = sdf_plane(p, pl);
        if (plane_dist < result.distance && plane_dist < 1000.0) {
            result.distance = plane_dist;
            result.object_type = 3u;
            result.object_index = i;
        }
    }
    
    return result;
}

// Simplified scene SDF for normal calculation
fn scene_sdf(p: vec3<f32>) -> f32 {
    return scene_sdf_detailed(p).distance;
}

// Structure to return ray marching result with object information
struct RayResult {
    t: f32,
    object_type: u32,
    object_index: u32,
}

// Ray marching with object tracking
fn ray_march_detailed(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> RayResult {
    // Start just in front of the camera near-plane
    var t = 0.01;
    let max_steps = 128;
    let max_dist  = 100.0;

    // March the ray through the scene SDF
    for (var i = 0; i < max_steps; i++) {
        let p = ray_origin + t * ray_dir;
        let scene_result = scene_sdf_detailed(p);

        // Adaptive convergence threshold – scales with travelled distance
        let min_dist = 0.0005 * t + 0.0005; // keeps precision near and far

        if (scene_result.distance < min_dist) {
            // --- Binary refinement to reduce overshoot --------------------
            var refine_t   = t;
            var refine_res = scene_result;
            for (var k = 0; k < 2; k++) {
                refine_t  -= refine_res.distance * 0.5;
                refine_res = scene_sdf_detailed(ray_origin + refine_t * ray_dir);
            }
            return RayResult(refine_t, refine_res.object_type, refine_res.object_index);
        }

        // Advance by full SDF distance
        t += scene_result.distance;
        if (t > max_dist) {
            break;
        }
    }

    // No hit
    return RayResult(-1.0, 0u, 0u);
}

// Simple ray marching for compatibility
fn ray_march(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    return ray_march_detailed(ray_origin, ray_dir).t;
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
    
    // Ray march with object information
    let ray_result = ray_march_detailed(ray_origin, ray_dir);
    
    if (ray_result.t < 0.0) {
        // Background
        return vec4<f32>(0.1, 0.2, 0.3, 1.0);
    }
    
    // Hit something
    let hit_pos = ray_origin + ray_result.t * ray_dir;
    let normal = calc_normal(hit_pos);
    
    // Simple lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let light = max(dot(normal, light_dir), 0.1);
    
    // Color based on object type and material properties
    var color = vec3<f32>(0.8, 0.8, 0.8); // Default gray
    
    if (ray_result.object_type == 0u) { // Sphere
        let sphere = spheres[ray_result.object_index];
        // Color based on material properties
        let friction_color = sphere.friction; // 0.0 to 1.0
        let restitution_color = sphere.restitution; // 0.0 to 1.0
        
        // High friction = red tint, low friction = blue tint
        // High restitution = green tint, low restitution = darker
        color = vec3<f32>(
            0.5 + friction_color * 0.5,      // Red channel: friction
            0.3 + restitution_color * 0.7,   // Green channel: restitution
            0.5 - friction_color * 0.3       // Blue channel: inverse friction
        );
    } else if (ray_result.object_type == 1u) { // Box
        color = vec3<f32>(0.8, 0.6, 0.4); // Brown for boxes
    } else if (ray_result.object_type == 2u) { // Cylinder
        color = vec3<f32>(0.6, 0.8, 0.6); // Green for cylinders
    } else if (ray_result.object_type == 3u) { // Plane
        color = vec3<f32>(0.7, 0.7, 0.8); // Light blue-gray for planes
    }
    
    // Apply lighting
    color *= light;
    
    return vec4<f32>(color, 1.0);
}