use crate::{BufferView, ComputeError};

/// Placeholder handler for ball joint solving.
pub fn handle_solve_ball_joints(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.is_empty() {
        return Err(ComputeError::ShapeMismatch("SolveBallJoints expects at least one buffer"));
    }
    Ok(vec![binds[0].data.to_vec()])
}
