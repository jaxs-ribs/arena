use crate::{BufferView, ComputeError};

pub fn handle_detect_contacts_sdf(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if !binds.is_empty() {
        Ok(vec![binds[0].data.to_vec()])
    } else {
        Ok(Vec::new())
    }
}
