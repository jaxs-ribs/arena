pub const STORAGE_IN: u32 = 0;
pub const STORAGE_IN2: u32 = 1; // binary ops
pub const STORAGE_OUT: u32 = 2;
pub const UNIFORM_SC: u32 = 3; // where mask or scalars

const _: () = assert!(STORAGE_OUT == 2);

/// Return expected number of bindings for each kernel.
pub const fn binding_count(kernel: &crate::Kernel) -> u32 {
    match kernel {
        crate::Kernel::SphereStep => 2,
        crate::Kernel::Add | crate::Kernel::Mul | crate::Kernel::Where => 4,
        crate::Kernel::ReduceSum => 3,
        crate::Kernel::MatMul => 3,
    }
}
