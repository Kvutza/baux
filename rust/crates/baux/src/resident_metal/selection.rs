//! Translate search state into the Metal proposal/selection ABI.
//!
//! This module packs observations, fitted ENN parameters, dissimilarities,
//! procedural seeds, and radius coefficients; it does not evaluate a model.
//! Historical `latent` names describe a geometry/storage mode, not a latent
//! parameterization of the network. That mode can use sampled parameter rows
//! through `Neighbors`, or procedural moment approximations when unavailable.
//! Moment estimates below are not exact rounded displacement measurements.

use super::*;

impl SearchState {
    #[cfg(test)]
    /// Build single-candidate kernel parameters for parity tests, carrying both
    /// the raw root and derived stream so program-specific seed mapping is
    /// tested.
    pub fn params(&self, root: u64, candidate: usize) -> Params {
        let radius = self.radius(candidate);
        Params {
            seed: self.procedural_seed(root, candidate),
            stream_seed: root,
            radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            ..Params::default()
        }
    }
    /// Pack pool-generation parameters without submitting GPU work.
    /// Preserves the historical fresh-stream mapping for the unspecialized
    /// non-rotation pool. `base_slot = u32::MAX` means no resident incumbent
    /// slot; it is not a valid history index.
    pub fn pool_params(&self, root: u64) -> Params {
        Params {
            seed: self.procedural_seed(root, 0),
            stream_seed: root,
            radius: self.radius(0),
            tiles: self.tiles.len() as u64 as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            blocks: self.blocks.len() as u32,
            base_slot: self.resident_identities[..self.resident_history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .and_then(|slot| u32::try_from(slot).ok())
                .unwrap_or(u32::MAX),
            ..Params::default()
        }
    }
    /// Snapshot observations, geometry, draws, and fitted scales for selection.
    /// Fitted ENN values override the supplied defaults; self-tuning computes
    /// each retained row's neighbor distance scale. History identities map
    /// resident storage to logical observations. `None` uses sentinel 4 for
    /// normal acquisition; the kernel, not this packer, chooses a candidate.
    pub fn select_params(
        &self,
        root: u64,
        mut config: RoundConfig,
        forced_candidate: Option<usize>,
    ) -> SelectionParams {
        if let Some((neighbors, epistemic, aleatoric, y_scale)) = self.fitted_enn {
            config.surrogate.neighbors = neighbors.min(self.history);
            config.surrogate.epistemic_scale = epistemic;
            config.surrogate.aleatoric_scale = aleatoric;
            config.surrogate.output_scale = y_scale;
        }
        let base_index = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)
            .unwrap_or(0);
        let base_distances = std::array::from_fn(|i| {
            if i < self.history {
                self.pairwise_distances[base_index * MAX_HISTORY + i]
            } else {
                0.0
            }
        });
        let local_scales = compute_scales(self);
        let resident_indices = resident_map(self);
        SelectionParams {
            root_seed: root,
            outcomes: std::array::from_fn(|i| self.outcomes.get(i).copied().unwrap_or(0.0)),
            variances: std::array::from_fn(|i| self.variances.get(i).copied().unwrap_or(0.0)),
            draws: std::array::from_fn(|i| {
                let identity = self.identities.get(i).copied().unwrap_or(0);
                crate::hash::normal_metric(config.acquisition.seed, identity, 0) as f32
            }),
            base_distances,
            local_scales,
            axis_history: std::array::from_fn(|i| {
                self.axis.as_ref().map_or(0.0, |axis| axis.history[i])
            }),
            axis_candidates: std::array::from_fn(|candidate| {
                self.axis_candidate(root, candidate).unwrap_or(0.0)
            }),
            axis_base: self.axis.as_ref().map_or(0.0, |axis| axis.base),
            axis_weight: self.axis.as_ref().map_or(0.0, |axis| axis.weight),
            axis_enabled: u32::from(self.axis.is_some()),
            epistemic_scale: config.surrogate.epistemic_scale,
            aleatoric_scale: config.surrogate.aleatoric_scale,
            y_scale: config.surrogate.output_scale,
            beta: config.acquisition.beta,
            radius: self.radius(0),
            neighbors: config.surrogate.neighbors as u32,
            history: self.history as u32,
            acquisition: match config.acquisition.kind {
                AcquisitionKind::Ucb => 0,
                AcquisitionKind::Thompson => 1,
                AcquisitionKind::Pareto => 2,
            },
            tiles: self.tiles.len() as u32,
            mode: self.perturbation.shader(),
            resident_history: self.physical_history() as u32,
            resident_indices,
            implicit_history: u32::from(self.implicit_history),
            forced_candidate: forced_candidate.unwrap_or(4) as u32,
            distance_metric: match self.distance_metric {
                crate::config::DistanceMetric::Euclidean => 0,
                crate::config::DistanceMetric::Conformal => 1,
                crate::config::DistanceMetric::Hyperbolic => 2,
                crate::config::DistanceMetric::Wasserstein => 3,
            },
            local_scale_neighbors: self.local_scale_neighbors as u32,
            incumbent_index: base_index as u32,
            candidate_floor: 0,
            adaptive_neighbors: u32::from(self.adapt_neighbors),
            pad: 0,
        }
    }

    /// Full parameter rows available to kernels, not the logical observation
    /// count. Implicit history can retain more observations than resident rows.
    pub fn physical_history(&self) -> usize {
        if self.implicit_history {
            self.resident_history
        } else {
            self.history
        }
    }
    /// Dimensionless multiplier applied to each block's proposal scale.
    /// Clamped to configured bounds. This is not the Euclidean norm of the
    /// displacement.
    pub fn radius(&self, _candidate: usize) -> f32 {
        self.length
            .clamp(self.length_config.length_min, self.length_config.length_max) as f32
    }
}

fn compute_scales(state: &SearchState) -> [f32; MAX_HISTORY] {
    let mut local_scales = [1.0; MAX_HISTORY];
    if state.distance_metric != crate::config::DistanceMetric::Euclidean && state.history > 1 {
        let mut values = Vec::with_capacity(state.history - 1);
        for (row, scale) in local_scales[..state.history].iter_mut().enumerate() {
            values.clear();
            values.extend(
                (0..state.history)
                    .filter(|&column| column != row)
                    .map(|column| state.total_distance(row, column))
                    .filter(|&d| d > 0.0),
            );
            if values.is_empty() {
                *scale = 1.0e-12;
            } else {
                values.sort_by(f32::total_cmp);
                *scale = values[state.local_scale_neighbors.min(values.len()) - 1].max(1.0e-12);
            }
        }
    }
    local_scales
}

fn resident_map(state: &SearchState) -> [u32; MAX_HISTORY] {
    std::array::from_fn(|i| {
        state.identities[..state.history]
            .iter()
            .position(|identity| {
                i < state.resident_history && *identity == state.resident_identities[i]
            })
            .map_or(u32::MAX, |index| index as u32)
    })
}
