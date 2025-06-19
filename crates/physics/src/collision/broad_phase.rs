//! Broad-phase collision detection using spatial partitioning

use crate::types::{Vec3, Sphere, SpatialGrid, BoundingBox};

/// Update spatial grid with current sphere positions
pub fn update_spatial_grid(grid: &mut SpatialGrid, spheres: &[Sphere]) {
    grid.clear();
    
    for (index, sphere) in spheres.iter().enumerate() {
        let bounds = sphere_bounding_box(sphere);
        grid.insert(index, bounds);
    }
}

/// Get potential collision pairs from spatial grid
pub fn get_potential_collision_pairs(grid: &SpatialGrid) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut checked = std::collections::HashSet::new();
    
    for (index_a, cell_indices) in grid.get_all_objects() {
        for cell_index in cell_indices {
            let neighbors = grid.get_cell_objects(cell_index);
            
            for &index_b in neighbors {
                if index_a < index_b {
                    let pair = (index_a, index_b);
                    if checked.insert(pair) {
                        pairs.push(pair);
                    }
                }
            }
        }
    }
    
    pairs
}

/// Calculate bounding box for a sphere
fn sphere_bounding_box(sphere: &Sphere) -> BoundingBox {
    let radius_vec = Vec3::new(sphere.radius, sphere.radius, sphere.radius);
    BoundingBox {
        min: sphere.pos - radius_vec,
        max: sphere.pos + radius_vec,
    }
}

/// Check if two bounding boxes overlap
pub fn boxes_overlap(a: &BoundingBox, b: &BoundingBox) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x &&
    a.min.y <= b.max.y && a.max.y >= b.min.y &&
    a.min.z <= b.max.z && a.max.z >= b.min.z
}