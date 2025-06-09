use phenotype::Phenotype;
use std::fs;

#[test]
fn parse_chain_example() {
    let json = fs::read_to_string("tests/data/chain.json").unwrap();
    let p = Phenotype::from_str(&json).unwrap();
    assert_eq!(p.bodies.len(), 3);
    assert_eq!(p.joints.len(), 2);
}

#[test]
fn sim_from_chain_runs() {
    let json = fs::read_to_string("tests/data/chain.json").unwrap();
    let p = Phenotype::from_str(&json).unwrap();
    let mut sim = p.into_sim().unwrap();
    sim.run_cpu(0.01, 5);
    assert_eq!(sim.spheres.len(), 3);
}

#[test]
fn parse_box_cylinder() {
    let json = fs::read_to_string("tests/data/box_cylinder.json").unwrap();
    let p = Phenotype::from_str(&json).unwrap();
    assert_eq!(p.bodies.len(), 2);
    assert_eq!(p.joints.len(), 1);
}

#[test]
fn parse_plane_sphere() {
    let json = fs::read_to_string("tests/data/sphere_plane.json").unwrap();
    let p = Phenotype::from_str(&json).unwrap();
    assert_eq!(p.bodies.len(), 2);
    assert!(p.joints.is_empty());
}
