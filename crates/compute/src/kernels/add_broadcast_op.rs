use crate::{BufferView, ComputeError};

/// Adds a vector to each row of a matrix.
///
/// Bindings `[a, b, output_placeholder]` expect `a` shaped `[batch, dim]` and
/// `b` shaped `[dim]`. The broadcasted sum is returned in a single buffer.
pub fn handle_add_broadcast(binds: &[BufferView]) -> Result<Vec<Vec<u8>>, ComputeError> {
    if binds.len() < 3 {
        return Err(ComputeError::ShapeMismatch(
            "AddBroadcast kernel expects 3 buffers (a, b, output_placeholder)",
        ));
    }
    let a_view = &binds[0];
    let b_view = &binds[1];
    let output_view = &binds[2];

    if a_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || b_view.element_size_in_bytes != std::mem::size_of::<f32>()
        || output_view.element_size_in_bytes != std::mem::size_of::<f32>()
    {
        return Err(ComputeError::ShapeMismatch(
            "AddBroadcast kernel currently only supports f32",
        ));
    }

    let a_data: &[f32] = bytemuck::cast_slice(&a_view.data);
    let b_data: &[f32] = bytemuck::cast_slice(&b_view.data);
    let mut out_data = vec![0.0f32; output_view.shape.iter().product()];

    let a_shape = &a_view.shape;
    let batch = a_shape[0];
    let dim = a_shape[1];

    for b_idx in 0..batch {
        for i in 0..dim {
            out_data[b_idx * dim + i] = a_data[b_idx * dim + i] + b_data[i];
        }
    }

    let out_bytes = bytemuck::cast_slice(&out_data).to_vec();
    Ok(vec![out_bytes])
} 
