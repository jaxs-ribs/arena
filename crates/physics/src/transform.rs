//! Transform matrix utilities for physics objects
//!
//! This module provides utilities for converting between physics representations
//! (position + quaternion) and 4x4 transformation matrices used by the renderer.

use crate::types::Vec3;
use glam::{Mat4, Quat, Vec3 as GlamVec3};

/// Convert position and quaternion to a 4x4 transformation matrix
pub fn to_transform_matrix(position: Vec3, orientation: [f32; 4]) -> [[f32; 4]; 4] {
    // Convert to glam types
    let pos = GlamVec3::new(position.x, position.y, position.z);
    let quat = Quat::from_xyzw(orientation[0], orientation[1], orientation[2], orientation[3]);
    
    // Create transform matrix
    let transform = Mat4::from_rotation_translation(quat, pos);
    
    // Convert to array format for GPU
    transform.to_cols_array_2d()
}

/// Convert position, quaternion, and scale to a 4x4 transformation matrix
pub fn to_transform_matrix_scaled(position: Vec3, orientation: [f32; 4], scale: Vec3) -> [[f32; 4]; 4] {
    // Convert to glam types
    let pos = GlamVec3::new(position.x, position.y, position.z);
    let quat = Quat::from_xyzw(orientation[0], orientation[1], orientation[2], orientation[3]);
    let scl = GlamVec3::new(scale.x, scale.y, scale.z);
    
    // Create transform matrix
    let transform = Mat4::from_scale_rotation_translation(scl, quat, pos);
    
    // Convert to array format for GPU
    transform.to_cols_array_2d()
}

/// Convert position and orientation to a 4x4 transform matrix with mesh offset
/// Used for objects where the visual mesh origin is offset from the body center
pub fn to_transform_matrix_with_offset(
    position: Vec3, 
    orientation: [f32; 4], 
    mesh_offset: Vec3
) -> [[f32; 4]; 4] {
    // Convert to glam types
    let pos = GlamVec3::new(position.x, position.y, position.z);
    let quat = Quat::from_xyzw(orientation[0], orientation[1], orientation[2], orientation[3]);
    let offset = GlamVec3::new(mesh_offset.x, mesh_offset.y, mesh_offset.z);
    
    // Transform: T(pos) * R(quat) * T(mesh_offset)
    // This applies the mesh offset in local space, then rotation and position
    let transform = Mat4::from_translation(pos) * 
                   Mat4::from_quat(quat) * 
                   Mat4::from_translation(offset);
    
    transform.to_cols_array_2d()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identity_transform() {
        let pos = Vec3::ZERO;
        let orientation = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
        
        let transform = to_transform_matrix(pos, orientation);
        
        // Should be identity matrix
        assert_eq!(transform[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(transform[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(transform[3], [0.0, 0.0, 0.0, 1.0]);
    }
    
    #[test]
    fn test_translation_only() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let orientation = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
        
        let transform = to_transform_matrix(pos, orientation);
        
        // Translation should be in the last column
        assert_eq!(transform[3], [1.0, 2.0, 3.0, 1.0]);
    }
    
    #[test]
    fn test_mesh_offset_transform() {
        let pos = Vec3::new(0.0, 1.0, 0.0);
        let orientation = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
        let offset = Vec3::new(0.0, -0.5, 0.0); // Mesh origin at bottom
        
        let transform = to_transform_matrix_with_offset(pos, orientation, offset);
        
        // With identity rotation, this should translate by pos + offset
        assert_eq!(transform[3][0], 0.0);
        assert_eq!(transform[3][1], 0.5); // 1.0 + (-0.5)
        assert_eq!(transform[3][2], 0.0);
    }
}