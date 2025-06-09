#![deny(clippy::all, clippy::pedantic)]

use anyhow::Result;
use physics::{PhysicsSim, Vec3};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Phenotype {
    pub bodies: Vec<Body>,
    #[serde(default)]
    pub joints: Vec<JointDef>,
}

#[derive(Deserialize)]
#[serde(tag = "shape")]
pub enum Body {
    #[serde(rename = "sphere")]
    Sphere {
        id: String,
        radius: f32,
        pos: [f32; 3],
        #[serde(default = "zero_vec")] 
        vel: [f32; 3],
    },
    #[serde(rename = "box")]
    Box {
        id: String,
        half_extents: [f32; 3],
        pos: [f32; 3],
        #[serde(default = "zero_vec")] 
        vel: [f32; 3],
    },
    #[serde(rename = "cylinder")]
    Cylinder {
        id: String,
        radius: f32,
        height: f32,
        pos: [f32; 3],
        #[serde(default = "zero_vec")] 
        vel: [f32; 3],
    },
    #[serde(rename = "plane")]
    Plane {
        id: String,
        normal: [f32; 3],
        d: f32,
    },
}

#[derive(Deserialize)]
pub struct JointDef {
    pub body_a: String,
    pub body_b: String,
    pub rest_length: f32,
}

fn zero_vec() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

impl Phenotype {
    #[must_use]
    pub fn from_str(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn into_sim(self) -> Result<PhysicsSim> {
        let mut sim = PhysicsSim::new();
        let mut map: HashMap<String, usize> = HashMap::new();
        for body in self.bodies {
            match body {
                Body::Sphere { id, radius: _, pos, vel } => {
                    let idx = sim.add_sphere(Vec3::new(pos[0], pos[1], pos[2]), Vec3::new(vel[0], vel[1], vel[2]));
                    map.insert(id, idx);
                }
                Body::Box { id, half_extents, pos, vel } => {
                    let idx = sim.add_box(
                        Vec3::new(pos[0], pos[1], pos[2]),
                        Vec3::new(half_extents[0], half_extents[1], half_extents[2]),
                        Vec3::new(vel[0], vel[1], vel[2]),
                    );
                    map.insert(id, idx);
                }
                Body::Cylinder { id, radius, height, pos, vel } => {
                    let idx = sim.add_cylinder(
                        Vec3::new(pos[0], pos[1], pos[2]),
                        radius,
                        height,
                        Vec3::new(vel[0], vel[1], vel[2]),
                    );
                    map.insert(id, idx);
                }
                Body::Plane { id, normal, d } => {
                    let idx = sim.add_plane(Vec3::new(normal[0], normal[1], normal[2]), d);
                    map.insert(id, idx);
                }
            }
        }

        for joint in self.joints {
            let a = map
                .get(&joint.body_a)
                .ok_or_else(|| anyhow::anyhow!("unknown body {}", joint.body_a))?;
            let b = map
                .get(&joint.body_b)
                .ok_or_else(|| anyhow::anyhow!("unknown body {}", joint.body_b))?;
            sim.add_joint(*a as u32, *b as u32, joint.rest_length);
        }

        Ok(sim)
    }
}
