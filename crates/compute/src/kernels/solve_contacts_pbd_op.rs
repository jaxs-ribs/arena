use crate::{BufferView, ComputeError};

pub fn handle_solve_contacts_pbd(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if !binds.is_empty() {
        Ok(vec![binds[0].data.to_vec()])
    } else {
        Ok(Vec::new())
    }
}
