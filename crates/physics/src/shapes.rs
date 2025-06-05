use crate::Vec3;

#[derive(Copy, Clone, Debug)]
pub struct Cube {
    pub side: f32,
    pub mass: f32,
}

impl Cube {
    #[must_use]
    pub const fn new(side: f32, mass: f32) -> Self {
        Self { side, mass }
    }

    /// Signed distance from point `p` to the cube centered at the origin.
    #[must_use]
    pub fn sdf(&self, p: Vec3) -> f32 {
        let half = self.side * 0.5;
        let qx = p.x.abs() - half;
        let qy = p.y.abs() - half;
        let qz = p.z.abs() - half;
        let dx = qx.max(0.0);
        let dy = qy.max(0.0);
        let dz = qz.max(0.0);
        let outside = (dx * dx + dy * dy + dz * dz).sqrt();
        let inside = qx.max(qy.max(qz)).min(0.0);
        outside + inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_sdf_at_center() {
        let c = Cube::new(2.0, 1.0);
        assert!((c.sdf(Vec3::new(0.0, 0.0, 0.0)) + 1.0).abs() < 1e-6);
    }
}

