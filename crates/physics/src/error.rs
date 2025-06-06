use compute::ComputeError;

#[derive(Debug)]
pub enum PhysicsError {
    BackendError(ComputeError),
    NoSpheres,
    // Other physics-specific errors can be added here
}

impl From<ComputeError> for PhysicsError {
    fn from(err: ComputeError) -> Self {
        PhysicsError::BackendError(err)
    }
} 