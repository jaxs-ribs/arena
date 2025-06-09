use physics::{PhysicsSim, Sphere, Vec3, Joint};
use crate::rl::Env;

/// Environment for balancing a stick by applying a horizontal force to the base sphere.
///
/// This is a placeholder for future reinforcement learning experiments. It models
/// two spheres connected by a distance joint. Gravity will cause the upper sphere
/// to fall unless the agent applies forces to keep it upright.
pub struct StickBalanceEnv {
    sim: PhysicsSim,
    base_idx: usize,
    tip_idx: usize,
}

impl StickBalanceEnv {
    /// Creates a new environment with a vertical stick of length 1.
    #[must_use]
    pub fn new() -> Self {
        let mut sim = PhysicsSim::new_single_sphere(0.0);
        // Add the second sphere representing the tip of the stick.
        sim.spheres.push(Sphere::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
        // Each sphere needs a force slot.
        sim.params.forces.push([0.0, 0.0]);
        // Constrain them with a distance joint so the stick maintains length.
        sim.joints.push(Joint { body_a: 0, body_b: 1, rest_length: 1.0, _padding: 0 });
        Self { sim, base_idx: 0, tip_idx: 1 }
    }

    /// Resets the environment with the tip offset by the given angle in radians.
    /// A small angle will cause the pole to fall over when no control is applied.
    pub fn reset_with_angle(&mut self, angle: f32) -> Vec<f32> {
        self.sim.spheres[self.base_idx].pos = Vec3::new(0.0, 0.0, 0.0);
        self.sim.spheres[self.base_idx].vel = Vec3::new(0.0, 0.0, 0.0);
        self.sim.spheres[self.tip_idx].pos = Vec3::new(-angle.sin(), angle.cos(), 0.0);
        self.sim.spheres[self.tip_idx].vel = Vec3::new(0.0, 0.0, 0.0);
        vec![0.0, angle]
    }

    /// Returns the angle of the stick relative to the vertical axis.
    fn stick_angle(&self) -> f32 {
        let base = &self.sim.spheres[self.base_idx];
        let tip = &self.sim.spheres[self.tip_idx];
        let dx = tip.pos.x - base.pos.x;
        let dy = tip.pos.y - base.pos.y;
        dy.atan2(dx) - std::f32::consts::FRAC_PI_2
    }
}

impl Env for StickBalanceEnv {
    fn step(&mut self, action: f32) -> (Vec<f32>, f32, bool) {
        // clamp horizontal force
        let force = action.max(-10.0).min(10.0);
        self.sim.params.forces[self.base_idx][0] = force;
        // advance physics by one step
        let _ = self.sim.step_gpu();

        let angle = self.stick_angle();
        let done = angle.abs() > std::f32::consts::FRAC_PI_4;
        // reward +1 for staying within angle limits
        let reward = if done { 0.0 } else { 1.0 };
        (vec![self.sim.spheres[self.base_idx].pos.x, angle], reward, done)
    }

    fn reset(&mut self) -> Vec<f32> {
        self.sim.spheres[self.base_idx].pos = Vec3::new(0.0, 0.0, 0.0);
        self.sim.spheres[self.base_idx].vel = Vec3::new(0.0, 0.0, 0.0);
        self.sim.spheres[self.tip_idx].pos = Vec3::new(0.0, 1.0, 0.0);
        self.sim.spheres[self.tip_idx].vel = Vec3::new(0.0, 0.0, 0.0);
        vec![0.0, 0.0]
    }

    fn obs_size(&self) -> usize { 2 }

    fn action_size(&self) -> usize { 1 }
}

