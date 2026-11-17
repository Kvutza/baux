//! Configure dynamic procedural candidate pool buffers and seed layout.

use super::*;
use crate::procedural_pool::{ProceduralPool, ProceduralPoolConfig};

#[cfg(test)]
mod tests;

impl SearchState {
    /// Resize dynamic pool scratch buffers and update arm failure-tolerance
    /// alignment (`ceil(max(4/arms, D/arms))` when not explicitly overridden)
    /// for arbitrary `pool.count()` (`1..=5000+`).
    pub fn configure_pool(mut self, pool: ProceduralPool) -> Result<Self, String> {
        self.check_idle()?;
        ProceduralPoolConfig {
            arms: pool.arms(),
            candidates_per_arm: pool.slots(),
        }
        .resolve_resident()?;
        if self.started || self.objective_selector.is_some() {
            return Err(
                "Configure procedural layout before asking or configuring vector acquisition"
                    .into(),
            );
        }
        if pool == self.pool_layout {
            return Ok(self);
        }
        let plan = initialization::allocation_sizes(
            self.dimensions,
            self.blocks.len(),
            self.tiles.len(),
            self.blocks.len() + 1,
            self.history_rows.len(),
            pool.count() as usize,
        )?;
        let additional = plan.resident_bytes.saturating_sub(self.resident_bytes);
        preflight(&self.runtime, &plan.sizes, additional)?;
        self.partials = Arc::new(self.runtime.buffer::<Partial>(plan.partial_count));
        self.pool_aggregates = Arc::new(self.runtime.buffer::<Partial>(plan.aggregate_count));
        self.candidate_scores = Arc::new(self.runtime.buffer::<CandidateScore>(plan.cand_count));
        self.selected_indices_gpu = Arc::new(self.runtime.buffer::<u32>(plan.cand_count));
        self.axis_candidates_gpu = Arc::new(self.runtime.buffer::<f32>(plan.cand_count));
        self.pool_distances = Arc::new(self.runtime.buffer::<f32>(plan.distance_count));
        self.pool_geometry = Arc::new(self.runtime.buffer::<f32>(plan.geometry_count));
        self.resident_bytes = plan.resident_bytes;
        self.trust.set_arms(pool.arms() as usize);
        self.pool_layout = pool;
        Ok(self)
    }

    pub fn procedural_seed(&self, root: u64, candidate: usize) -> u64 {
        self.pool_layout
            .seed(root, self.pool_layout.identity(candidate as u32).unwrap())
            .unwrap()
    }

    pub fn procedural_source(&self, _pool: ProceduralPool) -> String {
        initialization::search_source(self.independent_fp16, self.perturbation)
    }
}
