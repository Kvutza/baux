//! Translate search state into the Metal proposal/selection ABI.

use super::*;

impl SearchState {
    #[cfg(test)]
    pub fn params(&self, root: u64, candidate: usize) -> Params {
        let radius = self.radius(candidate);
        Params {
            seed: self.procedural_seed(root, candidate),
            stream_seed: root,
            radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.sampler.shader(),
            blocks: self.blocks.len() as u32,
            pool_count: self.pool_layout.count(),
            pool_slots: self.pool_layout.slots(),
            dimensions: self.dimensions as u32,
            num_pert: self.num_pert as u32,
            sobol_offset: self.sobol_offset,
            lhd_points: self.lhd_points,
            box_bounded: u32::from(self.box_bounded),
            has_dim_meta: u32::from(self.manifold.is_some()),
            coord_system: self
                .manifold
                .as_ref()
                .map_or(0, |m| manifold::coord_code(m.config.coord_system)),
            ..Params::default()
        }
    }

    pub fn pool_params(&self, root: u64) -> Params {
        Params {
            seed: self.procedural_seed(root, 0),
            stream_seed: root,
            radius: self.radius(0),
            candidate: u32::MAX,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.sampler.shader(),
            blocks: self.blocks.len() as u32,
            base_slot: self.resident_identities[..self.resident_history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .and_then(|slot| u32::try_from(slot).ok())
                .unwrap_or(u32::MAX),
            pool_count: self.pool_layout.count(),
            pool_slots: self.pool_layout.slots(),
            dimensions: self.dimensions as u32,
            num_pert: self.num_pert as u32,
            sobol_offset: self.sobol_offset,
            lhd_points: self.lhd_points,
            box_bounded: u32::from(self.box_bounded),
            has_dim_meta: u32::from(self.manifold.is_some()),
            coord_system: self
                .manifold
                .as_ref()
                .map_or(0, |m| manifold::coord_code(m.config.coord_system)),
            ..Params::default()
        }
    }

    pub(super) fn upload_axis(&self, root: u64) {
        if self.axis.is_none() {
            return;
        }
        let count = self.pool_layout.count() as usize;
        unsafe {
            let dst = self.axis_candidates_gpu.contents().cast::<f32>();
            for candidate in 0..count {
                dst.add(candidate)
                    .write(self.axis_candidate(root, candidate).unwrap_or(0.0));
            }
        }
    }

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

        let scalar_y_bounds = self
            .manifold
            .as_ref()
            .and_then(|m| m.config.y_bounds.as_ref())
            .and_then(|b| b.first().copied());
        let (y_bounds_mode, y_lower, y_upper) =
            scalar_y_bounds.map_or((0, 0.0, 0.0), manifold::bounds_mode);
        let (affine_a, affine_b, affine_c) = self
            .manifold
            .as_ref()
            .and_then(|m| m.affine.as_ref())
            .filter(|aff| !aff.a.is_empty())
            .map_or((0.0, 1.0, 1.0), |aff| {
                (aff.a[0] as f32, aff.b[0] as f32, aff.c[0] as f32)
            });

        let mut outcomes = [0.0f32; MAX_HISTORY];
        let mut variances = [0.0f32; MAX_HISTORY];
        for i in 0..self.history.min(MAX_HISTORY) {
            let raw_y = self.outcomes[i];
            let raw_v = if self.noise_aware {
                self.variances[i]
            } else {
                0.0
            };
            let (wy, wv) = manifold::warp_observation(raw_y, raw_v, scalar_y_bounds);
            outcomes[i] = wy;
            variances[i] = wv;
        }

        SelectionParams {
            root_seed: root,
            outcomes,
            variances,
            draws: std::array::from_fn(|i| {
                let identity = self.identities.get(i).copied().unwrap_or(0);
                crate::hash::normal_metric(config.acquisition.seed, identity, 0) as f32
            }),
            base_distances,
            local_scales,
            axis_history: std::array::from_fn(|i| {
                self.axis.as_ref().map_or(0.0, |axis| axis.history[i])
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
            acquisition: if self.history_rows.is_empty() {
                3
            } else {
                match config.acquisition.kind {
                    AcquisitionKind::Ucb => 0,
                    AcquisitionKind::Thompson => 1,
                    AcquisitionKind::Pareto => 2,
                    AcquisitionKind::Random => 3,
                }
            },
            tiles: self.tiles.len() as u32,
            mode: self.sampler.shader(),
            resident_history: self.physical_history() as u32,
            resident_indices,
            implicit_history: u32::from(self.implicit_history),
            forced_candidate: forced_candidate
                .or_else(|| {
                    (self.history_rows.is_empty()
                        || config.acquisition.kind == AcquisitionKind::Random)
                        .then(|| {
                            (crate::hash::splitmix64(root ^ config.acquisition.seed) as usize)
                                % (self.pool_layout.count() as usize)
                        })
                })
                .map_or(u32::MAX, |idx| idx as u32),
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
            pool_count: self.pool_layout.count(),
            pool_slots: self.pool_layout.slots(),
            num_arms: self.pool_layout.arms(),
            noise_aware: u32::from(self.noise_aware),
            y_bounds_mode,
            y_lower,
            y_upper,
            affine_a,
            affine_b,
            affine_c,
            pad: 0,
        }
    }

    pub fn physical_history(&self) -> usize {
        if self.history_rows.is_empty() {
            1
        } else if self.implicit_history {
            self.resident_history
        } else {
            self.history
        }
    }

    pub fn has_surrogate(&self) -> bool {
        !self.history_rows.is_empty()
    }

    pub fn radius(&self, _candidate: usize) -> f32 {
        if self
            .manifold
            .as_ref()
            .is_some_and(|m| m.config.no_trust_region)
        {
            return 2.0;
        }
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
