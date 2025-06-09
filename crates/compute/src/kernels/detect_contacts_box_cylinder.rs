use crate::{BufferView, ComputeError};

/// CPU fallback for box-cylinder collision detection.
///
/// This placeholder implementation simply returns an empty contact buffer.
pub fn handle_detect_contacts_box_cylinder(
    binds: &[BufferView],
) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "DetectContactsBoxCylinder expects 3 buffers (boxes, cylinders, contacts)",
        ));
    }
    Ok(vec![Vec::new()])
}

#[cfg(feature = "cpu-tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeBackend, CpuBackend, Kernel};
    use std::sync::Arc;

    #[test]
    fn dispatch_empty_ok() {
        let cpu = CpuBackend::new();
        let empty: Arc<[u8]> = Vec::<u8>::new().into();
        let view = BufferView::new(empty.clone(), vec![0], 1);
        let result = cpu
            .dispatch(
                &Kernel::DetectContactsBoxCylinder,
                &[view.clone(), view.clone(), view],
                [1, 1, 1],
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }
}
