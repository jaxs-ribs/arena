use std::fs;
use std::path::Path;

fn validate_shader(path: &Path) {
    let src = fs::read_to_string(path).expect("read shader");
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).expect("wgsl validate");
}

#[test]
fn compile_sdf_shader() {
    let shader = Path::new("../../crates/render/src/sdf.wgsl");
    validate_shader(shader);
}
