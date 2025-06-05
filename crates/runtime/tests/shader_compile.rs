use compute;
use std::fs;
use std::path::Path; // This should be resolvable as `runtime` depends on `compute`

// Helper function to validate a WGSL shader file using naga
fn validate_wgsl_shader(shader_path_str: &str) {
    let shader_path = Path::new(shader_path_str);
    let shader_source = match fs::read_to_string(shader_path) {
        Ok(s) => s,
        Err(e) => panic!("Failed to read shader file {shader_path:?}: {e}"),
    };

    let module = match naga::front::wgsl::parse_str(&shader_source) {
        Ok(m) => m,
        Err(e) => {
            panic!(
                "WGSL parsing error in {shader_path:?}:\n{error_report}",
                error_report = e.emit_to_string(&shader_source)
            );
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    match validator.validate(&module) {
        Ok(_) => println!("Successfully parsed and validated {shader_path:?}"),
        Err(e) => {
            panic!(
                "WGSL validation error in {shader_path:?}:\n{error_report}",
                error_report = e.emit_to_string(&shader_source)
            );
        }
    }
}

#[test]
fn validate_noop_shader_compiles() {
    let _backend = compute::default_backend();
    validate_wgsl_shader("../../shaders/noop.wgsl");
}

#[test]
fn validate_integrate_euler_shader_compiles() {
    let _backend = compute::default_backend();
    validate_wgsl_shader("../../shaders/integrate_euler.wgsl");
}

#[test]
fn validate_elementwise_shader_compiles() {
    let _backend = compute::default_backend();
    validate_wgsl_shader("../../shaders/elementwise.wgsl");
}
