//! Extensions for the SpatialGrid type

use crate::types::{SpatialGrid, BoundingBox};
use std::collections::HashMap;

/// Statistics about the spatial grid
pub struct SpatialGridStats {
    pub occupied_cells: usize,
    pub total_entries: usize,
    pub average_entries_per_cell: f32,
}

impl SpatialGrid {
    /// Clear all cells in the grid
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }
    
    /// Insert an object with the given bounding box
    pub fn insert(&mut self, index: usize, bounds: BoundingBox) {
        // Find the range of cells this bounding box occupies
        let min_grid = self.world_to_grid(bounds.min);
        let max_grid = self.world_to_grid(bounds.max);
        
        // Insert into all overlapping cells
        for z in min_grid[2]..=max_grid[2] {
            for y in min_grid[1]..=max_grid[1] {
                for x in min_grid[0]..=max_grid[0] {
                    if let Some(cell_index) = self.grid_to_index([x, y, z]) {
                        self.cells[cell_index].push(index);
                    }
                }
            }
        }
    }
    
    /// Get all objects and their cell indices
    pub fn get_all_objects(&self) -> HashMap<usize, Vec<usize>> {
        let mut object_cells: HashMap<usize, Vec<usize>> = HashMap::new();
        
        for (cell_idx, cell) in self.cells.iter().enumerate() {
            for &object_idx in cell {
                object_cells.entry(object_idx)
                    .or_insert_with(Vec::new)
                    .push(cell_idx);
            }
        }
        
        object_cells
    }
    
    /// Get all objects in a specific cell
    pub fn get_cell_objects(&self, cell_index: usize) -> &[usize] {
        if cell_index < self.cells.len() {
            &self.cells[cell_index]
        } else {
            &[]
        }
    }
    
    /// Get statistics about the grid
    pub fn get_stats(&self) -> SpatialGridStats {
        let occupied_cells = self.cells.iter()
            .filter(|cell| !cell.is_empty())
            .count();
        
        let total_entries: usize = self.cells.iter()
            .map(|cell| cell.len())
            .sum();
        
        let average_entries_per_cell = if occupied_cells > 0 {
            total_entries as f32 / occupied_cells as f32
        } else {
            0.0
        };
        
        SpatialGridStats {
            occupied_cells,
            total_entries,
            average_entries_per_cell,
        }
    }
}