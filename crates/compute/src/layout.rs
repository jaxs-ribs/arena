/// Binding slot for the first input storage buffer.
pub const STORAGE_IN: u32 = 0;
/// Binding slot for the second input buffer used by binary operations.
pub const STORAGE_IN2: u32 = 1; // binary ops
/// Binding slot for the third input buffer used by ternary operations.
pub const STORAGE_IN3: u32 = 2; // ternary ops (e.g. clamp input)
/// Binding slot for the output storage buffer.
pub const STORAGE_OUT: u32 = 3;
/// Binding slot for uniform configuration data.
pub const UNIFORM_SC: u32 = 4; // config, params, masks etc.

const _: () = assert!(STORAGE_OUT == 3);

/// Returns the expected number of buffer bindings for a given kernel.
///
/// The layout is stable across backends and used when creating bind groups.
/// Values may evolve as kernels mature.
pub const fn binding_count(kernel: &crate::Kernel) -> u32 {
    match kernel {
        // Element-wise
        crate::Kernel::Add => 3,
        crate::Kernel::Sub => 4,
        crate::Kernel::Mul => 4,
        crate::Kernel::Div
        | crate::Kernel::Min
        | crate::Kernel::Max
        | crate::Kernel::Where => 4, // IN1, IN2, OUT, CONFIG

        crate::Kernel::Neg
        | crate::Kernel::Exp
        | crate::Kernel::Log
        | crate::Kernel::Sqrt
        | crate::Kernel::Rsqrt
        | crate::Kernel::Tanh
        | crate::Kernel::Relu
        | crate::Kernel::Sigmoid => 3, // IN1, OUT, CONFIG

        crate::Kernel::Clamp => 5, // IN_VAL, IN_MIN, IN_MAX, OUT, CONFIG

        // Reductions
        crate::Kernel::ReduceSum => 3,
        crate::Kernel::ReduceMean => 3,
        crate::Kernel::ReduceMax => 3,

        crate::Kernel::SegmentedReduceSum | crate::Kernel::ScatterAdd => 4, // DATA_IN, INDICES, OUT, CONFIG

        crate::Kernel::Gather => 4, // DATA_IN, INDICES, OUT, CONFIG (Provisional)

        // Linear algebra
        crate::Kernel::MatMul => 4, // IN_A, IN_B, OUT, CONFIG

        // Physics world passes
        crate::Kernel::IntegrateBodies => 3, // BODIES_INOUT, PARAMS_UNIFORM, FORCES_IN
        crate::Kernel::DetectContactsSphere => 2, // BODIES_IN, CONTACTS_OUT
        crate::Kernel::DetectContactsBox => 3, // BODIES_IN, BOX_IN, CONTACTS_OUT
        crate::Kernel::DetectContactsSDF => 3, // BODIES_IN, SDF_DATA_UNIFORM_OR_STORAGE, CONTACTS_OUT
        crate::Kernel::SolveContactsPBD => 3,  // BODIES_INOUT, CONTACTS_IN, PARAMS_UNIFORM
        crate::Kernel::SolveJointsPBD => 3, // BODIES_INOUT, JOINTS_INOUT, PARAMS_UNIFORM

        // Optional helpers
        crate::Kernel::ExpandInstances => 3, // IN, OUT, CONFIG
        crate::Kernel::RngNormal => 2,       // OUT, CONFIG (e.g. seeds/state) (Provisional)
        crate::Kernel::AddBroadcast => 4,    // A, B, OUT, CFG
    }
}
