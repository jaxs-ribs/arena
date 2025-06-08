use std::sync::Arc;
use compute::ComputeBackend;

use crate::error::PhysicsError;
use crate::steps::{self, contact, joint};
use crate::types::{Joint, JointParams, PhysParams, Sphere, SphereState, Vec3};

pub struct PhysicsSim {
    pub spheres: Vec<Sphere>, // Host-side copy of sphere data
    pub params: PhysParams,
    pub joints: Vec<Joint>,
    pub joint_params: JointParams,
    backend: Arc<dyn ComputeBackend>,
}

impl PhysicsSim {
    /// Creates a new simulation with a single sphere at a given initial height.
    #[must_use]
    pub fn new_single_sphere(initial_height: f32) -> Self {
        let sphere = Sphere {
            pos: Vec3::new(0.0, initial_height, 0.0),
            vel: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            _pad: [0; 3],
        };
        let spheres = vec![sphere];

        let params = PhysParams {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 0.01,
            _pad: [0; 3],
        };

        let backend = compute::default_backend();

        Self {
            spheres,
            params,
            joints: Vec::new(),
            joint_params: JointParams {
                compliance: 0.0,
                _pad: [0.0; 3],
            },
            backend,
        }
    }

    pub fn reset(&mut self) {
        let initial_height = 10.0; // Or retrieve from some initial state config
        let sphere = Sphere {
            pos: Vec3::new(0.0, initial_height, 0.0),
            vel: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            _pad: [0; 3],
        };
        self.spheres = vec![sphere];
    }

    pub fn step(&mut self) -> Result<(), PhysicsError> {
        if self.spheres.is_empty() {
            return Ok(());
        }

        steps::integration::integrate_bodies(&*self.backend, &mut self.spheres, &self.params)?;
        
        contact::detect_and_solve_contacts(&*self.backend, &mut self.spheres)?;

        joint::solve_joints(&*self.backend, &mut self.spheres, &self.joints, &self.joint_params)?;

        Ok(())
    }

    pub fn run(&mut self, dt: f32, steps: usize) -> Result<SphereState, PhysicsError> {
        self.params.dt = dt;

        for _ in 0..steps {
            self.step()?;
        }

        self.spheres
            .get(0)
            .map(|s| SphereState { pos: s.pos })
            .ok_or(PhysicsError::NoSpheres)
    }
} 