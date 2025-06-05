// This test is expected to fail compilation until PhysicsSim is implemented.

// Import necessary items that might be used by PhysicsSim or the test later
use physics::PhysicsSim;

#[test]
fn sphere_free_fall_matches_analytic() {
    // initial height 10 m, no initial velocity
    let mut sim = PhysicsSim::new_single_sphere(10.0); // Made sim mutable
    let dt = 0.01_f32;          // 10 ms
    let steps = 100_usize;      // simulate 1 s so the sphere stays above ground
    let final_state = sim.run(dt, steps).unwrap();

    // analytic: h = h0 − ½ g t²  (g = 9.81)
    let expected = 10.0 - 0.5 * 9.81 * (dt * steps as f32).powi(2);
    let diff = (final_state.pos.y - expected).abs();
    assert!(diff < 1e-4, "diff={diff}"); // Corrected: Exact assert message
} 