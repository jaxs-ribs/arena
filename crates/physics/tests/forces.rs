/*
use physics::PhysicsSim;
use ml::{Dense, Tensor, Graph};

#[test]
fn zero_force_network_behaves_like_free_fall() {
    let mut sim = PhysicsSim::new_single_sphere(10.0);
    let dense = Dense::new(vec![0.0; 8], vec![0.0;2], 4, 2);
    let mut g = Graph::default();

    let baseline = {
        let mut s2 = PhysicsSim::new_single_sphere(10.0);
        s2.run(0.01, 100).unwrap().pos.y
    };

    for _ in 0..100 {
        let sphere = &sim.spheres[0];
        let input = Tensor::from_vec(vec![4], vec![sphere.pos.x, sphere.pos.y, sphere.vel.x, sphere.vel.y]);
        let out = dense.forward(&input, &mut g);
        sim.params.force = [out.data[0], out.data[1]];
        sim.step_gpu().unwrap();
    }

    assert!((sim.spheres[0].pos.y - baseline).abs() < 1e-4);
}
*/
