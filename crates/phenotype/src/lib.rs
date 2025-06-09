#![deny(clippy::all, clippy::pedantic)]
//! Provides types for describing procedurally generated agents.
//!
//! The data structures in this crate can be deserialized from a simple JSON
//! schema and later converted into a [`PhysicsSim`] provided by the `physics`
//! crate.  These types are mainly used by the runtime and machine learning
//! crates to build controllable creatures for simulation or training.

use anyhow::Result;
use physics::{PhysicsSim, Vec3};
use serde::Deserialize;
use std::collections::HashMap;

/// A fully specified creature description.
///
/// The `Phenotype` structure lists all bodies that make up the creature and
/// the joints that connect them.  It is typically loaded from a JSON file using
/// [`Phenotype::from_str`], after which it can be transformed into a running
/// [`PhysicsSim`] via [`Phenotype::into_sim`].
#[derive(Deserialize)]
pub struct Phenotype {
    /// Rigid bodies that compose the creature.
    pub bodies: Vec<Body>,
    /// Optional joint definitions. The field defaults to an empty list if it is
    /// omitted when deserializing.
    #[serde(default)]
    pub joints: Vec<JointDef>,
}

#[derive(Deserialize)]
#[serde(tag = "shape")]
/// An individual rigid body used in the simulation.
///
/// Bodies are tagged with the `shape` field when deserializing from JSON.  Each
/// variant stores the data necessary to create the corresponding shape in the
/// physics engine.
pub enum Body {
    /// A spherical body.
    #[serde(rename = "sphere")]
    Sphere {
        /// Unique identifier used by joints to reference this body.
        id: String,
        /// Sphere radius in metres.
        radius: f32,
        /// Initial position of the body.
        pos: [f32; 3],
        /// Linear velocity at the start of the simulation. Defaults to zero
        /// when omitted.
        #[serde(default = "zero_vec")]
        vel: [f32; 3],
    },
    /// An axis-aligned box body.
    #[serde(rename = "box")]
    Box {
        /// Unique identifier used by joints to reference this body.
        id: String,
        /// Half-extents of the box in each axis.
        half_extents: [f32; 3],
        /// Initial position of the body.
        pos: [f32; 3],
        /// Linear velocity at the start of the simulation. Defaults to zero
        /// when omitted.
        #[serde(default = "zero_vec")]
        vel: [f32; 3],
    },
    /// A cylinder aligned to the Y axis.
    #[serde(rename = "cylinder")]
    Cylinder {
        /// Unique identifier used by joints to reference this body.
        id: String,
        /// Cylinder radius.
        radius: f32,
        /// Cylinder height along the Y axis.
        height: f32,
        /// Initial position of the body.
        pos: [f32; 3],
        /// Linear velocity at the start of the simulation. Defaults to zero
        /// when omitted.
        #[serde(default = "zero_vec")]
        vel: [f32; 3],
    },
    /// An infinite plane.
    #[serde(rename = "plane")]
    Plane {
        /// Unique identifier used by joints to reference this body.
        id: String,
        /// Unit normal of the plane.
        normal: [f32; 3],
        /// The `d` coefficient of the plane equation `n * x + d = 0`.
        d: f32,
    },
}

#[derive(Deserialize)]
/// Description of a distance joint connecting two bodies.
pub struct JointDef {
    /// Identifier of the first body.
    pub body_a: String,
    /// Identifier of the second body.
    pub body_b: String,
    /// The rest length of the joint.
    pub rest_length: f32,
}

/// Helper used during deserialization to populate missing velocity fields.
fn zero_vec() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

impl Phenotype {
    /// Deserialize a [`Phenotype`] from its JSON representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is not valid or is missing required
    /// fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use phenotype::Phenotype;
    /// let json = r#"{"bodies": []}"#;
    /// let p = Phenotype::from_str(json).unwrap();
    /// assert!(p.bodies.is_empty());
    /// ```
    #[must_use]
    pub fn from_str(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Convert the description into a [`PhysicsSim`] ready to be executed.
    ///
    /// # Errors
    ///
    /// Returns an error if any joint references a body that does not exist in
    /// this phenotype.
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
