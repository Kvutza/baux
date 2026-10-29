//! Establish the resident optimizer's policies before observations and asks.
//!
//! Normal setup is `configure_enn`, optional `configure_controller`, then
//! `observe_initial`. Proposal program, noise sampler, history storage,
//! fitting, and radius control are separate choices. Unsupported combinations
//! are errors, not requests for a silent fallback to another optimizer.
//!
//! `latent_history` changes retained geometry/storage, not the number of model
//! parameters being optimized. Full rows/replay scratch are released in that
//! mode; supported FP16 innovation programs additionally retain sampled
//! parameter rows for neighbor distances. These samples are not full models.

use super::*;
use crate::params::ENNParams;

impl SearchState {
    /// Supply the initial scalar incumbent reward and estimate variance.
    /// Higher reward is better; pretraining passes negative mean NLL. Delegates
    /// initial-history/controller bookkeeping to `observe_objectives`; this
    /// method neither evaluates weights nor generates a proposal.
    pub(crate) fn observe_initial(&mut self, value: f32, variance: f32) -> Result<(), String> {
        self.observe_objectives(
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Set failure tolerance while idle and before any published ask.
    /// Updates the ordinary trust-region controller and any configured MORBO
    /// controller; it does not set reliability evidence thresholds or a radius.
    pub(crate) fn set_tolerance(&mut self, failures: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started {
            return Err("Set Metal BF16 failure tolerance before any asks".into());
        }
        self.trust
            .set_tolerance(failures)
            .map_err(|error| error.to_string())?;
        if let Some(morbo) = self
            .objective_selector
            .as_mut()
            .and_then(|selector| selector.morbo.as_mut())
        {
            morbo.set_tolerance(failures)?;
        }
        Ok(())
    }

    /// Install proposal, history, fitting, and query-neighbor policies.
    /// Requires idle implicit history with no observations and a valid ask.
    /// Sets warm-up count to configured k, prepares the fitter, and compiles
    /// the pool. Latent mode releases full history storage; compatible FP16
    /// Gaussian and Rademacher programs then allocate sampled parameter
    /// geometry. Pair rotation and threshold programs have explicit
    /// compatibility guards. Fitting variance scales and query-local
    /// neighbor adaptation are separate flags. Configuration can
    /// allocate/compile and is not rollback-atomic on failure; do not reuse
    /// a partially configured state after an error.
    pub(crate) fn configure_enn(
        &mut self,
        config: crate::config::ResidentEnnConfig,
    ) -> Result<(), String> {
        self.check_idle()?;
        check_ask(config.ask)?;
        if matches!(
            config.proposal_method,
            crate::procedural_pool::ProposalMethod::PairMix
                | crate::procedural_pool::ProposalMethod::QuadraticFlow
        ) && (!self.independent_fp16
            || config.history_geometry != crate::config::HistoryGeometry::Latent)
        {
            return Err(
                "Lie and pair-rotation require FP16 Gaussian or Rademacher proposals and sampled \
                 parameter history"
                    .into(),
            );
        }
        self.proposal_method = config.proposal_method;
        self.configure_pool(config.pool)?;
        if !self.implicit_history || self.history != 0 {
            return Err("Configure implicit ENN before its first observation".into());
        }
        if config.ask.surrogate.neighbors > MAX_HISTORY {
            return Err(format!(
                "Implicit ENN k={} exceeds logical history capacity {MAX_HISTORY}",
                config.ask.surrogate.neighbors
            ));
        }
        let k = i32::try_from(config.ask.surrogate.neighbors)
            .map_err(|_| "Implicit ENN k does not fit i32")?;
        let mut fitter = ENNFitter::new(k, true);
        fitter.set_params(
            ENNParams::new(
                k,
                f64::from(config.ask.surrogate.epistemic_scale),
                f64::from(config.ask.surrogate.aleatoric_scale),
            )
            .map_err(|error| error.to_string())?,
        );
        self.initial_observations = config.ask.surrogate.neighbors;
        self.fit_candidates = config.num_candidates;
        self.fit_samples = config.num_samples;
        self.fit_params = config.fit_params;
        self.adapt_neighbors = config.adapt_neighbors;
        self.fit_seed = config.ask.acquisition.seed;
        self.distance_scaling = config.distance_scaling;
        let was_latent = self.latent_history;
        self.latent_history = config.history_geometry == crate::config::HistoryGeometry::Latent;
        if self.latent_history && !was_latent {
            self.release_history();
        }
        self.local_scale_neighbors = config.local_scale_neighbors;
        self.fitter = Some(fitter);
        if self.latent_history && self.independent_fp16 {
            let neighbors = neighbors::Neighbors::new(self)?;
            self.resident_bytes += neighbors.bytes();
            self.neighbors = Some(neighbors);
        }
        Ok(())
    }

    /// Drop full historical rows and exact-replay storage for latent geometry.
    /// Keeps an incumbent alias for slot-based APIs, resets resident identity
    /// bookkeeping, substitutes minimal scratch buffers, and adjusts byte
    /// counts. Logical objective/distance storage remains; sampled
    /// geometry, when supported, is installed separately later in
    /// `configure_enn`.
    fn release_history(&mut self) {
        let mut released = 0u64;
        for row in &self.history_rows {
            if !std::ptr::eq::<metal::BufferRef>(&**row, &*self.base) {
                released = released.saturating_add(row.length());
            }
        }
        if let Some(origin) = self.replay_origin.take() {
            released = released.saturating_add(origin.length());
        }
        for buffer in [
            &self.replay_scales,
            &self.replay_partials,
            &self.replay_components,
            &self.pool_family_distances,
        ] {
            released = released.saturating_add(buffer.length());
        }
        self.history_rows.clear();
        self.history_rows.push(self.base.clone());
        self.anchor = self.base.clone();
        self.rejected = self.base.clone();
        self.resident_history = 0;
        self.resident_identities.fill(1);
        self.replay_scales = self.runtime.buffer::<f32>(1);
        self.replay_partials = self.runtime.buffer::<f32>(1);
        self.replay_components = self.runtime.buffer::<f32>(1);
        self.pool_family_distances = self.runtime.buffer::<f32>(1);
        self.exact_history = false;
        let retained = self.replay_scales.length()
            + self.replay_partials.length()
            + self.replay_components.length()
            + self.pool_family_distances.length();
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(released)
            .saturating_add(retained);
    }

    /// Install reliability radius control after ENN configuration, before data.
    /// Requires idle implicit history with no observations or published asks.
    /// Validates the policy using existing length bounds; it does not measure
    /// reliability, fit ENN, or change the proposal direction formula.
    pub(crate) fn configure_controller(
        &mut self,
        config: reliability_region::ReliabilityPolicy,
    ) -> Result<(), String> {
        self.check_idle()?;
        if !self.implicit_history || self.started || self.history != 0 {
            return Err("Configure reliability control before the initial observation".into());
        }
        if self.fitter.is_none() {
            return Err("Configure the resident ENN before reliability control".into());
        }
        self.reliability = Some(reliability_region::ReliabilityController::new(
            config,
            self.length_config,
        )?);
        Ok(())
    }

    /// Last fitted `(k, epistemic_scale, aleatoric_scale, y_scale)`, if
    /// available. Reading this telemetry does not fit parameters or select
    /// a local k.
    pub(crate) fn fitted_enn(&self) -> Option<(usize, f32, f32, f32)> {
        self.fitted_enn
    }

    /// Enable learned tensor-family metric weights and proposal scales once.
    /// Requires idle, unobserved implicit FP16 state and validated block
    /// groups. Allocates fitting scratch and writes group/base-weight
    /// metadata; this is family-wise scaling, not a learned dense parameter
    /// covariance matrix.
    pub(crate) fn enable_family(&mut self, groups: Vec<usize>) -> Result<(), String> {
        self.check_idle()?;
        if !self.independent_fp16
            || !self.implicit_history
            || self.history != 0
            || self.family.is_some()
        {
            return Err("Configure learned family shape once before initial observation".into());
        }
        let family = FamilyHistory::new(groups, &self.blocks)?;
        let metric = metric::MetricGpu::new(Arc::clone(&self.runtime))?;
        self.resident_bytes = self
            .resident_bytes
            .checked_add(metric.bytes())
            .ok_or("Metric scratch memory accounting overflow")?;
        unsafe {
            for (index, (&group, block)) in family.groups.iter().zip(&family.original).enumerate() {
                self.family_groups
                    .contents()
                    .cast::<u32>()
                    .add(index)
                    .write(group as u32);
                self.family_base_weights
                    .contents()
                    .cast::<f32>()
                    .add(index)
                    .write(block.weight);
            }
        }
        self.family = Some(family);
        self.metric_gpu = Some(metric);
        Ok(())
    }

    /// Current family metric weights and proposal-scale multipliers, or `None`
    /// for a static shape. Read-only; these are not full parameter-space axes.
    pub(crate) fn family_shape(&self) -> Option<([f32; FAMILIES], [f32; FAMILIES])> {
        self.family.as_ref().map(|f| (f.weights, f.scales()))
    }

    /// Current count of recorded history slots.
    pub fn history(&self) -> usize {
        self.history
    }

    /// Recompute block scales for GPU leaves.
    pub(super) fn sync_scales(&mut self) -> Result<(), String> {
        let Some(family) = &self.family else {
            return Ok(());
        };
        let scales = family.scales();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let g = family.groups[i];
            block.scale = family.original[i].scale * scales[g];
            block.weight = family.original[i].weight * family.weights[g];
        }
        let (_, _, leaves) = make_layout(&self.blocks, self.dimensions)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                leaves.as_ptr(),
                self.leaves_gpu.contents().cast::<Leaf>(),
                leaves.len(),
            );
        }
        Ok(())
    }

    /// Apply the selected family shape to block scales, GPU leaves, and
    /// distances. Called after completed evaluator/tell work, with no ask
    /// reading the shared leaves. Rebuilds aggregate history distances from
    /// family components; a no-op when family learning is disabled. Does
    /// not regenerate model weights.
    pub(super) fn apply_family(&mut self) -> Result<(), String> {
        self.sync_scales()?;
        let Some(family) = &self.family else {
            return Ok(());
        };
        let distances = family.aggregate(family.weights, self.history);
        for i in 0..self.history {
            for j in 0..self.history {
                self.pairwise_distances[i * MAX_HISTORY + j] = distances[[i, j]] as f32;
            }
        }
        Ok(())
    }

    /// Enable a seeded correlated reference once on fresh non-FP16 search.
    /// Requires idle state before asks; model-sized allocation is deferred to
    /// `prepare_reference` or the first ask. Rejected for independent FP16
    /// mode.
    pub fn correlate(&mut self, reference_seed: u64) -> Result<(), String> {
        self.check_idle()?;
        if self.independent_fp16 || self.started || self.reference_seed.is_some() {
            return Err("Enable correlated Metal sampling once before any asks".into());
        }
        self.reference_seed = Some(reference_seed);
        Ok(())
    }

    /// Eagerly build/check the optional reference while idle.
    /// Moves its allocation/GPU work outside a later ask's latency measurement;
    /// a no-op for independent FP16. Does not propose or evaluate a trial.
    pub(crate) fn prepare_reference(&mut self) -> Result<(), String> {
        self.check_idle()?;
        self.ensure_ref()
    }

    /// Switch fresh correlated two-row search to paired-relative outcomes.
    /// Requires an existing observation/reference, no asks, and tolerance four;
    /// clears stored rewards/variances and makes absolute tell APIs
    /// incompatible. This mode is distinct from passing an absolute paired
    /// measurement to `paired_modeled`; it does not itself evaluate either
    /// member of a pair.
    pub fn enable_relative(&mut self, failure_tolerance: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started
            || self.relative
            || self.history == 0
            || self.reference_seed.is_none()
            || failure_tolerance != 4
            || self.history_rows.len() != 2
        {
            return Err(
                "Metal paired-relative mode requires fresh correlated search and \
                 failure_tolerance=4"
                    .into(),
            );
        }
        self.relative = true;
        self.outcomes.fill(0.0);
        self.variances.fill(0.0);
        Ok(())
    }
}
