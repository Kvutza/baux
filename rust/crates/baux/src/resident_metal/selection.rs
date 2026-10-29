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
    /// Kernel program tag: 3 for pair rotation, 4 for quadratic flow, otherwise
    /// 0 for the ordinary innovation path. This tag is separate from the
    /// Gaussian/Rademacher sampler mode.
    pub(super) fn proposal_program(&self) -> u32 {
        match self.proposal_method {
            crate::procedural_pool::ProposalMethod::QuadraticFlow => 4,
            crate::procedural_pool::ProposalMethod::PairMix => 3,
            _ => 0,
        }
    }

    /// Whether full realized/replay geometry, rather than latent-mode geometry,
    /// is enabled. This flag does not assert metric axioms or prediction
    /// accuracy.
    pub(super) fn realized_history(&self) -> bool {
        self.exact_history && !self.latent_history
    }

    /// Whether ask may skip materializing the full parameter-sized pool.
    /// Latent mode ranks using sampled parameter geometry when installed,
    /// threshold geometry, or procedural approximations. Only the selected row
    /// is fully materialized; `true` does not mean every distance is analytic
    /// or that the selected row is subsequently scored without a forward pass.
    pub(super) fn analytic_pool(&self) -> bool {
        self.latent_history
    }

    /// Procedural squared-norm estimate: sum of `len * weight * (scale*r)^2`.
    /// Equals the ideal second moment for independent, centered, unit-variance
    /// additive innovations. It ignores rounding and sample cross terms and is
    /// NOT the exact norm of the state-dependent pair-rotation program. Used
    /// as procedural geometry/metadata, separately from sampled distances.
    pub(super) fn latent_norm(&self, candidate: usize) -> f32 {
        let radius = f64::from(self.radius(candidate));
        self.blocks
            .iter()
            .map(|block| {
                let step = f64::from(block.scale) * radius;
                block.len as f64 * f64::from(block.weight) * step * step
            })
            .sum::<f64>() as f32
    }

    /// Split the same ideal innovation norm into configured tensor families.
    /// Uses original family metric weights with current block scales. Returns
    /// zeros if family learning is disabled; not a measurement of realized
    /// pairs.
    pub(super) fn latent_components(&self, candidate: usize) -> [f32; FAMILIES] {
        let mut norms = [0.0f64; FAMILIES];
        let Some(family) = &self.family else {
            return [0.0; FAMILIES];
        };
        let radius = f64::from(self.radius(candidate));
        for (index, block) in self.blocks.iter().enumerate() {
            let step = f64::from(block.scale) * radius;
            norms[family.groups[index]] +=
                block.len as f64 * f64::from(family.original[index].weight) * step * step;
        }
        norms.map(|norm| norm as f32)
    }

    #[cfg(test)]
    /// Build single-candidate kernel parameters for parity tests, carrying both
    /// the raw root and derived stream so program-specific seed mapping is
    /// tested.
    pub(super) fn params(&self, root: u64, candidate: usize) -> Params {
        let radius = self.radius(candidate);
        Params {
            seed: self.procedural_seed(root, candidate),
            stream_seed: root,
            basis_seed: 0,
            radius,
            alternate_radius: radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            program: self.proposal_program(),
            ..Params::default()
        }
    }
    /// Pack pool-generation parameters without submitting GPU work.
    /// Preserves the historical fresh-stream mapping for the unspecialized
    /// non-rotation pool. `base_slot = u32::MAX` means no resident incumbent
    /// slot; it is not a valid history index.
    pub(super) fn pool_params(&self, root: u64) -> Params {
        Params {
            seed: self.procedural_seed(root, 0),
            stream_seed: if self.pool_layout == crate::procedural_pool::ProceduralPool::legacy()
                && self.proposal_method != crate::procedural_pool::ProposalMethod::PairMix
                && self.proposal_method != crate::procedural_pool::ProposalMethod::QuadraticFlow
            {
                candidate_seed(root, 2)
            } else {
                root
            },
            basis_seed: 0,
            radius: self.radius(0),
            alternate_radius: self.radius(1),
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            program: self.proposal_program(),
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
    pub(super) fn select_params(
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
            if self.implicit_history && i < self.history {
                self.pairwise_distances[base_index * MAX_HISTORY + i]
            } else {
                0.0
            }
        });
        let local_scales = compute_scales(self);
        let resident_indices = resident_map(self);
        SelectionParams {
            root_seed: root,
            basis_seed: 0,
            outcomes: std::array::from_fn(|i| self.outcomes.get(i).copied().unwrap_or(0.0)),
            variances: std::array::from_fn(|i| self.variances.get(i).copied().unwrap_or(0.0)),
            draws: std::array::from_fn(|i| {
                let identity = if self.relative {
                    self.identities
                        .get(i)
                        .copied()
                        .filter(|&id| id != 0)
                        .unwrap_or_else(|| i as i64 + 1)
                } else {
                    self.identities.get(i).copied().unwrap_or(0)
                };
                crate::hash::normal_metric(config.acquisition.seed, identity, 0) as f32
            }),
            base_distances,
            local_scales,
            latent_norms: std::array::from_fn(|candidate| self.latent_norm(candidate)),
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
            alternate_radius: self.radius(1),
            neighbors: config.surrogate.neighbors as u32,
            history: self.history as u32,
            acquisition: match config.acquisition.kind {
                AcquisitionKind::Ucb => 0,
                AcquisitionKind::Thompson => 1,
                AcquisitionKind::Pareto => 2,
            },
            tiles: self.tiles.len() as u32,
            mode: self.perturbation.shader(),
            program: self.proposal_program(),
            resident_history: self.physical_history() as u32,
            resident_indices,
            implicit_history: u32::from(self.implicit_history),
            exact_history: u32::from(
                self.realized_history() && self.history > self.resident_history,
            ),
            latent_history: u32::from(self.latent_history),
            forced_candidate: forced_candidate.unwrap_or(4) as u32,
            distance_scaling: match self.distance_scaling {
                crate::config::DistanceScaling::Global => 0,
                crate::config::DistanceScaling::SelfTuning => 1,
                crate::config::DistanceScaling::Hyperbolic => 2,
            },
            local_scale_neighbors: self.local_scale_neighbors as u32,
            incumbent_index: base_index as u32,
            candidate_floor: 0,
            sampled_history: u32::from(self.neighbors.is_some()),
            adaptive_neighbors: u32::from(self.adapt_neighbors),
        }
    }

    /// Full parameter rows available to kernels, not the logical observation
    /// count. Implicit history can retain more observations than resident rows.
    pub(super) fn physical_history(&self) -> usize {
        if self.implicit_history {
            self.resident_history
        } else {
            self.history
        }
    }
    /// Dimensionless multiplier applied to each block's proposal scale.
    /// Single-slot pools use `self.length`, and multi-slot pools use 0.5x/2x
    /// `self.length` for even/odd arm indices, clamped to configured bounds.
    /// This is not the Euclidean norm of the displacement.
    pub(super) fn radius(&self, candidate: usize) -> f32 {
        (self.length
            * if self.pool_layout.slots() == 1 {
                1.0
            } else if candidate & 1 == 0 {
                0.5
            } else {
                2.0
            })
        .clamp(self.length_config.length_min, self.length_config.length_max) as f32
    }
    /// Ensure the optional correlated BF16 reference exists inside an
    /// autorelease pool; a no-op for the independent FP16 representation.
    pub(super) fn ensure_ref(&mut self) -> Result<(), String> {
        autoreleasepool(|| self.init_reference())
    }

    /// Lazily allocate and generate the configured correlated reference.
    /// Requires a reference seed outside FP16 mode, checks memory and per-block
    /// RMS, and waits for GPU completion. Drops the new reference on failure.
    pub(super) fn init_reference(&mut self) -> Result<(), String> {
        if self.independent_fp16 {
            return Ok(());
        }
        let seed = self
            .reference_seed
            .ok_or("Enable correlated Metal sampling first")?;
        if self.reference.is_none() {
            preflight(&self.runtime, &[self.row_bytes()], self.row_bytes())?;
            self.reference = Some(self.runtime.buffer::<u16>(self.dimensions));
            let command = self.runtime.queue.new_command_buffer();
            self.encode_ref(
                command,
                Params {
                    seed,
                    initialize: 1,
                    ..Params::default()
                },
            );
            if let Err(error) = finish(command).and_then(|()| self.check_reference()) {
                self.reference = None;
                return Err(error);
            }
        }
        Ok(())
    }
    /// Reject nonfinite or nonpositive stored reference RMS values.
    /// Checks scale metadata only, not objective quality or every reference
    /// word.
    pub(super) fn check_reference(&self) -> Result<(), String> {
        if read::<f32>(&self.reference_scales, self.blocks.len())
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            Err("Invalid Metal BF16 reference RMS".into())
        } else {
            Ok(())
        }
    }
}

fn compute_scales(state: &SearchState) -> [f32; MAX_HISTORY] {
    let mut local_scales = [1.0; MAX_HISTORY];
    if state.distance_scaling != crate::config::DistanceScaling::Global && state.history > 1 {
        let mut values = Vec::with_capacity(state.history - 1);
        for (row, scale) in local_scales[..state.history].iter_mut().enumerate() {
            values.clear();
            values.extend(
                (0..state.history)
                    .filter(|&column| column != row)
                    .map(|column| state.total_distance(row, column)),
            );
            values.sort_by(f32::total_cmp);
            *scale = values[state.local_scale_neighbors.min(values.len()) - 1].max(1.0e-12);
        }
    }
    local_scales
}

fn resident_map(state: &SearchState) -> [u32; 2] {
    std::array::from_fn(|i| {
        state.identities[..state.history]
            .iter()
            .position(|identity| {
                i < state.resident_history && *identity == state.resident_identities[i]
            })
            .unwrap_or(0) as u32
    })
}
