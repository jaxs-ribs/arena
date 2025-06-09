use crate::{BufferView, ComputeError};

/// CPU fallback for cylinder-cylinder collision detection.
///
/// This placeholder implementation simply returns no contacts.
pub fn handle_detect_contacts_cylinder_cylinder(
    binds: &[BufferView],
) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 2 {
        return Err(ComputeError::ShapeMismatch(
            "DetectContactsCylinderCylinder expects 2 buffers (cylinders, contacts)",
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
                &Kernel::DetectContactsCylinderCylinder,
                &[view.clone(), view],
                [1, 1, 1],
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }
}
