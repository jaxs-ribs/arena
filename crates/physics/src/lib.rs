#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)] // Allow PhysicsSim in physics crate

// Refined imports
use compute::{ComputeBackend, ComputeError}; // BufferView and Kernel will be fully qualified
use std::sync::Arc;
use std::mem::size_of; // Specific import for size_of

// --- Data Structures ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[must_use] 
pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    pub pos: Vec3,
    pub vel: Vec3,
}

// This struct will be passed to the shader as uniform.
// WGSL `params: vec4<f32>` where xyz: gravity, w: dt.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhysParams {
    pub gravity: Vec3, // Mapped to params.xyz in shader
    pub dt: f32,       // Mapped to params.w in shader
}

// Structure to return from sim.run() to satisfy the test
pub struct SphereState {
    pub pos: Vec3,
    // pub vel: Vec3, // If needed later
}

// --- Error Type ---

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

// --- Physics Simulator ---

pub struct PhysicsSim {
    spheres: Vec<Sphere>, // Host-side copy of sphere data
    params: PhysParams,
    backend: Arc<dyn ComputeBackend>, // Using Arc for flexibility, could be Box
                                      // The test calls sim.run() so sim needs to be mutable or run needs &mut self
                                      // If run is &mut self, then backend doesn't strictly need Arc unless sim is cloned
                                      // For now, let's assume backend can be shared or PhysicsSim is not cloned while a run is ongoing.
}

const WORKGROUP_SIZE: u32 = 256;

impl PhysicsSim {
    /// Creates a new simulation with a single sphere at a given initial height.
    #[must_use]
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let sphere = Sphere {
            pos: Vec3::new(0.0, initial_height, 0.0),
            vel: Vec3::new(0.0, 0.0, 0.0),
        };
        let spheres = vec![sphere];

        let params = PhysParams {
            gravity: Vec3::new(0.0, -9.81, 0.0), // Default gravity, will be used by step_gpu
            dt: 0.01, // Default dt, will be overridden by run method's dt
        };

        let backend = Arc::new(compute::MockCpu);

        Self {
            spheres,
            params,
            backend,
        }
    }

    pub fn step_gpu(&mut self) -> Result<(), PhysicsError> {
        if self.spheres.is_empty() {
            return Ok(());
        }

        let sphere_bytes_arc: Arc<[u8]> = bytemuck::cast_slice(&self.spheres).to_vec().into();
        let sphere_buffer_view = compute::BufferView::new(
            sphere_bytes_arc, // Pass the Arc directly
            vec![self.spheres.len()],
            size_of::<Sphere>(),
        );

        let params_bytes_arc: Arc<[u8]> = bytemuck::bytes_of(&self.params).to_vec().into();
        let params_buffer_view = compute::BufferView::new(
            params_bytes_arc, // Pass the Arc directly
            vec![1],
            size_of::<PhysParams>(),
        );

        let num_spheres = self.spheres.len() as u32;
        let workgroups_x = (num_spheres + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Call the updated dispatch method
        let output_data_vec = self.backend.dispatch(
            &compute::Kernel::SphereStep,
            &[sphere_buffer_view, params_buffer_view],
            [workgroups_x, 1, 1],
        )?;

        // Handle the output data
        if let Some(updated_sphere_bytes) = output_data_vec.get(0) { // Assuming first buffer is spheres
            if updated_sphere_bytes.len() == self.spheres.len() * size_of::<Sphere>() {
                // Safely cast the bytes back to spheres
                // bytemuck::try_cast_slice requires the slice to have the exact size and alignment.
                // A more robust way if alignment isn't guaranteed by Vec<u8> is to copy element by element
                // or use bytemuck::pod_read_unaligned for each element if needed.
                // However, Vec<Sphere> written via cast_slice and read back into Vec<u8> should be fine with cast_slice.
                match bytemuck::try_cast_slice::<u8, Sphere>(updated_sphere_bytes) {
                    Ok(updated_spheres_slice) => {
                        self.spheres.clone_from_slice(updated_spheres_slice);
                    }
                    Err(e) => {
                        // This would indicate a serious issue with data integrity or alignment
                        // For now, let's wrap it in a PhysicsError. A more specific error would be better.
                        eprintln!("Failed to cast sphere bytes from GPU: {e:?}");
                        return Err(PhysicsError::BackendError(ComputeError::ShapeMismatch(
                            "Failed to cast updated sphere data from GPU due to alignment or size mismatch"
                        )));
                    }
                }
            } else {
                // Size mismatch
                return Err(PhysicsError::BackendError(ComputeError::ShapeMismatch(
                    "Returned sphere data size does not match expected size",
                )));
            }
        } else {
            // No data returned, which is unexpected for SphereStep that modifies spheres
            return Err(PhysicsError::BackendError(ComputeError::ShapeMismatch(
                "No sphere data returned from GPU after SphereStep dispatch",
            )));
        }

        Ok(())
    }

    pub fn run(&mut self, dt: f32, steps: usize) -> Result<SphereState, PhysicsError> {
        if self.spheres.is_empty() {
            return Err(PhysicsError::NoSpheres);
        }
        self.params.dt = dt;

        for _ in 0..steps {
            self.step_gpu()?;
        }

        Ok(SphereState { pos: self.spheres[0].pos })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_single_sphere() {
        let sim = PhysicsSim::new_single_sphere(10.0);
        assert_eq!(sim.spheres.len(), 1, "Should have one sphere");
        let sphere = sim.spheres[0];
        assert_eq!(sphere.pos.x, 0.0);
        assert_eq!(sphere.pos.y, 10.0);
        assert_eq!(sphere.pos.z, 0.0);
        assert_eq!(sphere.vel.x, 0.0);
        assert_eq!(sphere.vel.y, 0.0);
        assert_eq!(sphere.vel.z, 0.0);

        // Check params (gravity is default, dt is default but will be set by run())
        assert_eq!(sim.params.gravity.y, -9.81);
        // assert_eq!(sim.params.dt, 0.01); // Initial dt before run()
    }

    // Unit test for step_gpu (will fail or do nothing until step_gpu is implemented)
    #[test]
    fn test_step_gpu_mock_ok_with_valid_buffers() {
        let mut sim = PhysicsSim::new_single_sphere(5.0);
        let result = sim.step_gpu();
        assert!(result.is_ok(), "step_gpu with MockCpu should return Ok if buffers are valid, got {result:?}");
        // Further assertions could check if MockCpu (if it modified anything) did correctly.
        // For now, MockCpu in compute crate only does shape checks and returns Ok if shapes are fine.
        // The actual step_gpu implementation will involve creating BufferViews and calling dispatch.
        // This test will become more meaningful once step_gpu does that.
    }

    // Unit test for run (will likely show sphere not moving until step_gpu is real)
    #[test]
    fn test_run_single_sphere_mock_does_not_move_sphere() {
        let mut sim = PhysicsSim::new_single_sphere(10.0);
        let dt = 0.01_f32;
        let steps = 100_usize;
        let final_state = sim.run(dt, steps).expect("run should succeed");
        // With MockCpu, the sphere's position won't change from its initial state
        // because MockCpu's dispatch does no computation.
        assert_eq!(final_state.pos.y, 10.0, "Sphere y-pos should not change with MockCpu");
    }
} 