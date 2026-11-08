//! Trust region implementations for TuRBO optimizer.

use ndarray::{Array1, ArrayView1, s};

mod null;
pub use null::{
    MorboTRSettings, NoTrustRegion, Rescalarize, TRLengthConfig, TrustRegionConfig,
    TrustRegionError, TrustRegionOutcome,
};

/// Trust region for single-objective optimization (TuRBO).
///
/// Implements success/failure-based length adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboTrustRegion {
    /// Current trust region length.
    length: f64,
    /// Consecutive failure count.
    failure_counter: i32,
    /// Consecutive success count.
    success_counter: i32,
    /// Best observed value.
    best_value: f64,
    /// Number of observations at last update.
    prev_obs: usize,
    /// Success tolerance (consecutive successes before expansion).
    success_tolerance: i32,
    /// Failure tolerance (consecutive failures before contraction).
    failure_tolerance: Option<i32>,
    /// Number of dimensions.
    num_dim: usize,
    /// Batch size (number of arms).
    num_arms: Option<usize>,
    /// Length configuration.
    config: TRLengthConfig,
    /// Optional effective dimension for failure tolerance (defaults to
    /// num_dim).
    failure_tolerance_dim: Option<f64>,
    failure_tolerance_override: Option<i32>,
    pub(crate) clamp_domain: bool,
}

impl TurboTrustRegion {
    /// Create a new TurboTrustRegion.
    pub fn new(num_dim: usize, config: TRLengthConfig) -> Self {
        Self {
            length: config.length_init,
            failure_counter: 0,
            success_counter: 0,
            best_value: f64::NEG_INFINITY,
            prev_obs: 0,
            success_tolerance: 3,
            failure_tolerance: None,
            num_dim,
            num_arms: None,
            config,
            failure_tolerance_dim: None,
            failure_tolerance_override: None,
            clamp_domain: true,
        }
    }

    /// Override the ambient dimension used for failure tolerance.
    pub fn set_dim(&mut self, dim: f64) {
        self.failure_tolerance_dim = Some(dim);
        self.compute_tolerance();
    }

    /// Initialize or update batch size.
    pub fn set_arms(&mut self, num_arms: usize) {
        self.num_arms = Some(num_arms);
        self.compute_tolerance();
    }

    /// Use an explicit failure budget, independent of dimension and batch size.
    pub fn set_tolerance(&mut self, failures: usize) -> Result<(), TrustRegionError> {
        let failures = i32::try_from(failures)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                TrustRegionError::InvalidParameter(
                    "failure tolerance must be in 1..=i32::MAX".into(),
                )
            })?;
        self.failure_tolerance_override = Some(failures);
        self.compute_tolerance();
        Ok(())
    }

    /// Compute failure tolerance based on num_arms and num_dim.
    fn compute_tolerance(&mut self) {
        if let Some(failures) = self.failure_tolerance_override {
            self.failure_tolerance = Some(failures);
            return;
        }
        if let Some(num_arms) = self.num_arms {
            let eff_dim = self.failure_tolerance_dim.unwrap_or(self.num_dim as f64);
            self.failure_tolerance = Some(baux::semantics::optimizer::turbo_tolerance(
                eff_dim, num_arms,
            ));
        }
    }

    fn adapt_step(&mut self, improved: Option<bool>) {
        let failure_tol = self.failure_tolerance();
        let _ = baux::semantics::optimizer::turbo_adapt(
            &mut self.length,
            self.config.length_max,
            &mut self.success_counter,
            &mut self.failure_counter,
            self.success_tolerance,
            failure_tol,
            improved,
        );
    }

    /// Get current trust region length.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Consecutive improvements required before the region expands.
    pub fn success_tolerance(&self) -> i32 {
        self.success_tolerance
    }

    /// Consecutive non-improvements required before the region contracts.
    pub fn failure_tolerance(&self) -> i32 {
        self.failure_tolerance.unwrap_or(4)
    }

    /// Current consecutive-improvement count.
    pub fn success_counter(&self) -> i32 {
        self.success_counter
    }

    /// Current consecutive-non-improvement count.
    pub fn failure_counter(&self) -> i32 {
        self.failure_counter
    }

    /// Observation count at last trust-region update.
    pub fn prev_obs(&self) -> usize {
        self.prev_obs
    }

    /// Morbo: set incumbent scalar before an update step.
    pub fn set_best(&mut self, value: f64) {
        self.best_value = value;
    }

    /// Morbo: update using zero-padded history and a scalar incumbent value
    /// (Python parity).
    pub fn update_scalar(
        &mut self,
        num_obs: usize,
        y_incumbent_value: f64,
        scale: f64,
    ) -> Result<(), TrustRegionError> {
        if num_obs == 0 || num_obs == self.prev_obs {
            return Ok(());
        }
        if num_obs < self.prev_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                num_obs, self.prev_obs
            )));
        }
        if !self.best_value.is_finite() {
            self.best_value = y_incumbent_value;
            self.prev_obs = num_obs;
            return Ok(());
        }
        let improved = y_incumbent_value > self.best_value + 1e-3 * scale;
        self.adapt_step(Some(improved));
        self.best_value = self.best_value.max(y_incumbent_value);
        self.prev_obs = num_obs;
        Ok(())
    }

    /// Check if restart is needed (length below minimum).
    pub fn needs_restart(&self) -> bool {
        self.length < self.config.length_min
    }

    /// Update trust region based on new observations.
    ///
    /// Matches Python: scale is computed from all observations before the
    /// current batch (y_all[0..prev_obs]), not just the current batch.
    ///
    /// # Arguments
    ///
    /// * `y_all` - All objective values so far (full observation history)
    /// * `num_obs` - Total number of observations (must equal y_all.len())
    pub fn update(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_obs {
            return Ok(());
        }
        if n < self.prev_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                n, self.prev_obs
            )));
        }
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal y_all.len() {}",
                num_obs, n
            )));
        }

        // First update: establish best value and return
        if !self.best_value.is_finite() {
            let new_best = y_all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.best_value = new_best;
            self.prev_obs = n;
            return Ok(());
        }

        // prev_values = observations before this batch (matches Python)
        let prev_slice = y_all.slice(s![..self.prev_obs]);
        let new_batch = y_all.slice(s![self.prev_obs..]);

        let new_best = new_batch.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Scale from prev_values (all observations before current batch)
        let prev_len = prev_slice.len();
        let scale = if prev_len >= 2 {
            let min_val = prev_slice.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = prev_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (max_val - min_val).max(1e-6)
        } else if prev_len == 0 {
            0.0
        } else {
            // Single value: max - min = 0 (matches Python)
            0.0
        };

        // Check for improvement
        let improvement_threshold = 1e-3 * scale;
        let improved = new_best > self.best_value + improvement_threshold;
        if improved {
            self.best_value = self.best_value.max(new_best);
        }
        self.adapt_step(Some(improved));

        self.prev_obs = n;
        Ok(())
    }

    /// Update trust region using observed history for scale and incumbent
    /// scalar for improvement.
    ///
    /// Matches Python `TurboTrustRegion.update(y_obs, y_incumbent)`.
    pub fn update_history(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_obs {
            return Ok(());
        }
        if n < self.prev_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                n, self.prev_obs
            )));
        }
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal y_all.len() {}",
                num_obs, n
            )));
        }

        let y_new = y_all.slice(s![self.prev_obs..]);
        self.update_batch(&y_new, num_obs, y_incumbent_value)
    }

    /// Same as [`Self::update_history`], but only the newly told batch is
    /// required.
    ///
    /// Maintains running y min/max so improvement scale is O(1) in history
    /// length.
    pub fn update_batch(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        update_sense(
            self,
            y_new,
            num_obs,
            y_incumbent_value,
            crate::sense::Maximize,
        )
    }

    /// Update from an explicit model-agreement outcome.
    ///
    /// Unlike [`Self::update_batch`], an inconclusive noisy observation neither
    /// expands nor contracts the region and preserves accumulated evidence.
    pub fn update_outcome(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
        outcome: TrustRegionOutcome,
    ) -> Result<(), TrustRegionError> {
        if y_new.is_empty() {
            return Ok(());
        }
        let n = self.prev_obs + y_new.len();
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal prev_obs {} + y_new.len() {}",
                num_obs,
                self.prev_obs,
                y_new.len()
            )));
        }
        if !y_incumbent_value.is_finite() {
            return Err(TrustRegionError::InvalidParameter(
                "incumbent value must be finite".into(),
            ));
        }

        if !self.best_value.is_finite() {
            self.best_value = y_incumbent_value;
            self.prev_obs = n;
            return Ok(());
        }

        let improved = match outcome {
            TrustRegionOutcome::Success => Some(true),
            TrustRegionOutcome::Failure => Some(false),
        };
        self.adapt_step(improved);

        if outcome == TrustRegionOutcome::Success {
            self.best_value = self.best_value.max(y_incumbent_value);
        }
        self.prev_obs = n;
        Ok(())
    }

    /// Compute trust region bounds in 1D.
    ///
    /// Returns (lower_bounds, upper_bounds) for each dimension.
    pub fn compute_bounds(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let num_dim = x_center.len();
        let half_length = if let Some(ls) = lengthscales {
            ls * self.length / 2.0
        } else {
            Array1::from_elem(num_dim, self.length / 2.0)
        };

        let lb = x_center - &half_length;
        let ub = x_center + &half_length;

        if self.clamp_domain {
            let lb = lb.mapv(|v| v.clamp(0.0, 1.0));
            let ub = ub.mapv(|v| v.clamp(0.0, 1.0));
            (lb, ub)
        } else {
            (lb, ub)
        }
    }

    /// Restart the trust region (reset to initial state).
    ///
    /// Keeps `prev_obs` so the next tell can use the O(batch) path without
    /// reloading full observation history (critical for disk-backed large N).
    pub fn restart(&mut self) {
        self.length = self.config.length_init;
        self.failure_counter = 0;
        self.success_counter = 0;
        self.best_value = f64::NEG_INFINITY;
    }

    /// Reset the complete local-region baseline after discarding its dataset.
    pub fn restart_local(&mut self) {
        self.restart();
        self.prev_obs = 0;
    }

    /// Set observation watermark without scanning history (post-init /
    /// restart).
    pub fn set_watermark(&mut self, prev_obs: usize) {
        self.prev_obs = prev_obs;
    }

    /// Seed baseline best value and observation watermark (e.g. after warmup).
    pub fn seed_baseline(&mut self, best_value: f64, prev_obs: usize) {
        if best_value.is_finite() {
            self.best_value = best_value;
        }
        self.prev_obs = prev_obs;
    }
}

/// Sense-parametric trust-region batch update supporting
/// [`crate::sense::Maximize`], [`crate::sense::Minimize`], and
/// [`crate::sense::Target`].
pub fn update_sense<O: crate::sense::ObjectiveSense>(
    tr: &mut TurboTrustRegion,
    y_new: &ArrayView1<f64>,
    num_obs: usize,
    y_incumbent_value: f64,
    sense: O,
) -> Result<(), TrustRegionError> {
    let n_new = y_new.len();
    if n_new == 0 {
        return Ok(());
    }
    let n = tr.prev_obs + n_new;
    if num_obs != n {
        return Err(TrustRegionError::InvalidParameter(format!(
            "num_obs {} must equal prev_obs {} + y_new.len() {}",
            num_obs, tr.prev_obs, n_new
        )));
    }

    let utility = sense.score(y_incumbent_value).raw();
    if !tr.best_value.is_finite() {
        tr.best_value = utility;
        tr.prev_obs = n;
        return Ok(());
    }

    let scale = improvement_scale(tr.best_value);
    let improved = utility > tr.best_value + 1e-3 * scale;
    tr.adapt_step(Some(improved));
    tr.best_value = tr.best_value.max(utility);
    tr.prev_obs = n;
    Ok(())
}

fn improvement_scale(best_value: f64) -> f64 {
    if best_value.is_finite() {
        best_value.abs()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests;
