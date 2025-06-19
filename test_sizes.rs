use physics::types::Sphere;

fn main() {
    println\!("Sphere size: {}", std::mem::size_of::<Sphere>());
    println\!("TestSphere size from kernel: {}", 64); // Based on the struct in integrate_bodies_op.rs
}
