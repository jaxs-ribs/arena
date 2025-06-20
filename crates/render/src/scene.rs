//! Scene management and GPU buffer updates
//!
//! This module handles updating GPU buffers with physics simulation data,
//! converting from physics types to GPU-compatible formats.

use crate::gpu_types::{BoxGpu, CylinderGpu, PlaneGpu, SceneCounts, SphereGpu};
use physics::{BoxBody, Cylinder, Plane, Sphere};

/// Scene buffer manager
///
/// Handles the conversion and upload of physics primitives to GPU buffers
pub struct SceneManager {
    /// Storage buffers for each primitive type
    pub spheres_buffer: wgpu::Buffer,
    pub boxes_buffer: wgpu::Buffer,
    pub cylinders_buffer: wgpu::Buffer,
    pub planes_buffer: wgpu::Buffer,
    /// Buffer containing primitive counts
    pub counts_buffer: wgpu::Buffer,
}

impl SceneManager {
    /// Update all scene buffers with new physics data
    ///
    /// Converts physics primitives to GPU format and uploads to the corresponding buffers
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        spheres: &[Sphere],
        boxes: &[BoxBody],
        cylinders: &[Cylinder],
        planes: &[Plane],
    ) {
        // Convert and upload spheres
        if !spheres.is_empty() {
            let sphere_data: Vec<SphereGpu> = spheres.iter().map(SphereGpu::from).collect();
            queue.write_buffer(
                &self.spheres_buffer,
                0,
                bytemuck::cast_slice(&sphere_data),
            );
        }

        // Convert and upload boxes
        if !boxes.is_empty() {
            let box_data: Vec<BoxGpu> = boxes.iter().map(BoxGpu::from).collect();
            queue.write_buffer(
                &self.boxes_buffer,
                0,
                bytemuck::cast_slice(&box_data),
            );
        }

        // Convert and upload cylinders
        if !cylinders.is_empty() {
            let cylinder_data: Vec<CylinderGpu> = cylinders.iter().map(CylinderGpu::from).collect();
            
            // DEBUG: Log what orientation data we're uploading to GPU
            let gpu_orientation = cylinder_data[0].orientation;
            tracing::debug!("🔍 GPU upload: cylinder orientation=[{:.3}, {:.3}, {:.3}, {:.3}]", 
                           gpu_orientation[0], gpu_orientation[1], gpu_orientation[2], gpu_orientation[3]);
            
            queue.write_buffer(
                &self.cylinders_buffer,
                0,
                bytemuck::cast_slice(&cylinder_data),
            );
        }

        // Convert and upload planes
        if !planes.is_empty() {
            let plane_data: Vec<PlaneGpu> = planes.iter().map(PlaneGpu::from).collect();
            queue.write_buffer(
                &self.planes_buffer,
                0,
                bytemuck::cast_slice(&plane_data),
            );
        }

        // Update counts
        let counts = SceneCounts {
            spheres: spheres.len() as u32,
            boxes: boxes.len() as u32,
            cylinders: cylinders.len() as u32,
            planes: planes.len() as u32,
        };
        queue.write_buffer(&self.counts_buffer, 0, bytemuck::bytes_of(&counts));
    }

    /// Clear all primitives from the scene
    pub fn clear(&self, queue: &wgpu::Queue) {
        let counts = SceneCounts {
            spheres: 0,
            boxes: 0,
            cylinders: 0,
            planes: 0,
        };
        queue.write_buffer(&self.counts_buffer, 0, bytemuck::bytes_of(&counts));
    }
}