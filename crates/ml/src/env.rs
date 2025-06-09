/// Reinforcement learning environment trait.
///
/// Inspired by classic frameworks like OpenAI Gym, this trait defines the core
/// interface an environment must provide. Each call to [`step`] advances the
/// simulation by one action and returns the new observation vector, a reward
/// signal, and whether the episode has terminated.
///
/// [`step`]: Env::step
pub trait Env {
    /// Advance the environment by one action.
    ///
    /// Returns `(obs, reward, done)` where `obs` is the new observation vector,
    /// `reward` is the scalar reward, and `done` indicates episode termination.
    fn step(&mut self, action: f32) -> (Vec<f32>, f32, bool);

    /// Reset the environment to its starting state and return the initial
    /// observation vector.
    fn reset(&mut self) -> Vec<f32>;

    /// Size of the observation vector.
    fn obs_size(&self) -> usize;

    /// Size of the action space.
    fn action_size(&self) -> usize;
}
