// Debug shader to visualize cylinder orientation

// Simple cylinder SDF without rotation - always vertical
fn sdf_cylinder_no_rotation(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    let offset = p - center;
    let d = vec2<f32>(length(offset.xz), abs(offset.y)) - vec2<f32>(radius, height * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// Cylinder with rotation - should show tilted cylinder
fn sdf_cylinder_with_rotation(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32, orientation: vec4<f32>) -> f32 {
    // Transform point to cylinder's local space by applying inverse rotation
    let offset = p - center;
    let qv = vec3<f32>(-orientation.x, -orientation.y, -orientation.z); // conjugate
    let qw = orientation.w;
    let local_p = offset + 2.0 * cross(qv, cross(qv, offset) + qw * offset);
    
    // Standard cylinder SDF in local space (cylinder aligned with Y-axis)
    let d = vec2<f32>(length(local_p.xz), abs(local_p.y)) - vec2<f32>(radius, height * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// Debug visualization - show orientation as color
fn debug_cylinder_color(orientation: vec4<f32>) -> vec3<f32> {
    // Extract angle from quaternion (assuming Z-axis rotation)
    let angle = 2.0 * asin(orientation.z);
    
    // Map angle to color
    // Green = upright (0°)
    // Red = tilted right (+angle)  
    // Blue = tilted left (-angle)
    let red = max(0.0, angle / 0.5);  // 0 to 1 for 0 to ~30°
    let blue = max(0.0, -angle / 0.5); // 0 to 1 for 0 to -30°
    let green = 1.0 - abs(angle) / 0.5; // 1 to 0 as angle increases
    
    return vec3<f32>(red, green, blue);
}