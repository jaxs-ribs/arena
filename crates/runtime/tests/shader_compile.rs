use compute;
use std::fs;
use std::path::Path;

fn validate_wgsl_shader(shader_path: &Path) {
    let shader_source = fs::read_to_string(shader_path)
        .expect(&format!("Failed to read shader file {:?}", shader_path));
    let module = naga::front::wgsl::parse_str(&shader_source)
        .expect(&format!("WGSL parsing error in {:?}", shader_path));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect(&format!("WGSL validation error in {:?}", shader_path));
}

#[test]
fn validate_kernel_shaders_compile() {
    let _backend = compute::default_backend();
    for entry in fs::read_dir(Path::new("../../shaders")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map(|e| e == "wgsl").unwrap_or(false) {
            validate_wgsl_shader(&path);
        }
    }
}
