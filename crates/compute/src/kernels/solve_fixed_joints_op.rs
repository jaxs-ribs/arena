use crate::{BufferView, ComputeError};

/// Placeholder handler for fixed joint solving.
pub fn handle_solve_fixed_joints(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.is_empty() {
        return Err(ComputeError::ShapeMismatch("SolveFixedJoints expects at least one buffer"));
    }
    Ok(vec![binds[0].data.to_vec()])
}
