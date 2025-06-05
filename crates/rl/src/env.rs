/*
/// Very simple environment where the optimal action equals a target value.
/// State is ignored and episodes last a single step.
#[derive(Clone)]
pub struct SimpleEnv {
    pub target: f32,
}

impl SimpleEnv {
    #[must_use]
    pub fn new(target: f32) -> Self { Self { target } }

    #[must_use]
    pub fn reset(&self) -> () { () }

    pub fn step(&self, _state: &mut (), action: f32) -> ((), f32, bool) {
        let reward = -(action - self.target).powi(2);
        ((), reward, true)
    }
}
*/
