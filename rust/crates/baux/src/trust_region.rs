//! Trust region implementations for TuRBO optimizer.

use deser::{Deserialize, Serialize};
use ndarray::{Array1, ArrayView1, s};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum Rescalarize {
    OnRestart,
    OnPropose,
}

impl std::str::FromStr for Rescalarize {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "on-restart" => Ok(Self::OnRestart),
            "on-propose" => Ok(Self::OnPropose),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MorboTRSettings {
    pub num_metrics: usize,
    pub alpha: f64,
    pub length: TRLengthConfig,
    pub rescalarize: Rescalarize,
    pub noise_aware: bool,
}

/// Trust-region configuration (TuRBO or Morbo).
#[derive(Debug, Clone)]
pub enum TrustRegionConfig {
    Turbo(TRLengthConfig),
    Morbo(MorboTRSettings),
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        TrustRegionConfig::Turbo(TRLengthConfig::default())
    }
}

/// Errors that can occur in trust region operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TrustRegionError {
    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Invalid state for operation.
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Evidence supplied by an objective evaluation about the local model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustRegionOutcome {
    Success,
    Failure,
    Inconclusive,
}

/// Configuration for trust region length parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRLengthConfig {
    /// Initial trust region length.
    pub length_init: f64,
    /// Minimum trust region length.
    pub length_min: f64,
    /// Maximum trust region length.
    pub length_max: f64,
}

impl Default for TRLengthConfig {
    fn default() -> Self {
        Self {
            length_init: 0.8,
            length_min: 0.5f64.powi(7), // 0.5^7 ≈ 0.0078
            length_max: 1.6,
        }
    }
}

impl TRLengthConfig {
    /// Create new TRLengthConfig with custom values.
    pub fn new(length_init: f64, length_min: f64, length_max: f64) -> Self {
        Self {
            length_init,
            length_min,
            length_max,
        }
    }

    /// Validated constructor enforcing finite bounds satisfying 0 < length_min
    /// <= length_init <= length_max.
    pub fn try_new(
        length_init: f64,
        length_min: f64,
        length_max: f64,
    ) -> Result<Self, TrustRegionError> {
        if !length_init.is_finite()
            || !length_min.is_finite()
            || !length_max.is_finite()
            || length_min <= 0.0
            || length_min > length_init
            || length_init > length_max
        {
            return Err(TrustRegionError::InvalidParameter(
                "Trust region lengths must be finite and satisfy 0 < min <= init <= max".into(),
            ));
        }
        Ok(Self::new(length_init, length_min, length_max))
    }

    /// Clamp a candidate radius within [length_min, length_max].
    #[inline]
    pub fn clamp_radius(&self, r: f64) -> f64 {
        if self.length_min.is_nan() || self.length_max.is_nan() {
            return f64::NAN;
        }
        let (low, high) = if self.length_min <= self.length_max {
            (self.length_min, self.length_max)
        } else {
            (self.length_max, self.length_min)
        };
        r.clamp(low, high)
    }
}

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
    /// Min of y[0..prev_obs] for O(1) improvement scale (no full-history scan).
    hist_ymin: f64,
    /// Max of y[0..prev_obs] for O(1) improvement scale.
    hist_ymax: f64,
    /// Optional effective dimension for failure tolerance (defaults to
    /// num_dim).
    failure_tolerance_dim: Option<f64>,
    failure_tolerance_override: Option<i32>,
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
            hist_ymin: f64::INFINITY,
            hist_ymax: f64::NEG_INFINITY,
            failure_tolerance_dim: None,
            failure_tolerance_override: None,
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
            let tolerance = ((4.0 / num_arms as f64).max(eff_dim / num_arms as f64)).ceil() as i32;
            self.failure_tolerance = Some(tolerance.max(1));
        }
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
        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }
        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);
        if self.success_counter >= success_tol {
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            self.length *= 0.5;
            self.failure_counter = 0;
        }
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
            self.success_counter += 1;
            self.failure_counter = 0;
            self.best_value = self.best_value.max(new_best);
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }

        // Adapt length
        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);

        if self.success_counter >= success_tol {
            // Expand: double length
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            // Contract: halve length
            self.length *= 0.5;
            self.failure_counter = 0;
        }

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
        let n_new = y_new.len();
        if n_new == 0 {
            return Ok(());
        }
        let n = self.prev_obs + n_new;
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal prev_obs {} + y_new.len() {}",
                num_obs, self.prev_obs, n_new
            )));
        }

        if !self.best_value.is_finite() {
            self.best_value = y_incumbent_value;
            incorporate_hist(&mut self.hist_ymin, &mut self.hist_ymax, y_new);
            self.prev_obs = n;
            return Ok(());
        }

        let scale = improvement_scale(self.best_value);
        let improved = y_incumbent_value > self.best_value + 1e-3 * scale;
        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }

        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);
        if self.success_counter >= success_tol {
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            self.length *= 0.5;
            self.failure_counter = 0;
        }
        self.best_value = self.best_value.max(y_incumbent_value);
        incorporate_hist(&mut self.hist_ymin, &mut self.hist_ymax, y_new);
        self.prev_obs = n;
        Ok(())
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

        match outcome {
            TrustRegionOutcome::Success => {
                self.success_counter += 1;
                self.failure_counter = 0;
            }
            TrustRegionOutcome::Failure => {
                self.failure_counter += 1;
                self.success_counter = 0;
            }
            TrustRegionOutcome::Inconclusive => {}
        }

        if self.success_counter >= self.success_tolerance {
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= self.failure_tolerance() {
            self.length *= 0.5;
            self.failure_counter = 0;
        }

        self.best_value = if self.best_value.is_finite() {
            self.best_value.max(y_incumbent_value)
        } else {
            y_incumbent_value
        };
        incorporate_hist(&mut self.hist_ymin, &mut self.hist_ymax, y_new);
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

        // Clip to [0, 1]
        let lb = lb.mapv(|v| v.clamp(0.0, 1.0));
        let ub = ub.mapv(|v| v.clamp(0.0, 1.0));

        (lb, ub)
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
        self.hist_ymin = f64::INFINITY;
        self.hist_ymax = f64::NEG_INFINITY;
    }

    /// Set observation watermark without scanning history (post-init /
    /// restart).
    pub fn set_watermark(&mut self, prev_obs: usize) {
        self.prev_obs = prev_obs;
    }

    /// Seed the scale and incumbent baseline from observations preceding the
    /// first length update.
    pub fn seed_history(&mut self, prefix: &ArrayView1<f64>) {
        incorporate_hist(&mut self.hist_ymin, &mut self.hist_ymax, prefix);
        if !self.best_value.is_finite() {
            if let Some(best) = prefix.iter().copied().reduce(f64::max) {
                self.best_value = best;
            }
        }
    }
}

fn incorporate_hist(hist_ymin: &mut f64, hist_ymax: &mut f64, y_new: &ArrayView1<f64>) {
    for &v in y_new.iter() {
        *hist_ymin = hist_ymin.min(v);
        *hist_ymax = hist_ymax.max(v);
    }
}

fn improvement_scale(best_value: f64) -> f64 {
    if best_value.is_finite() {
        best_value.abs()
    } else {
        0.0
    }
}

#[path = "trust_region/null.rs"]
mod null;
pub use null::NoTrustRegion;

#[cfg(test)]
#[path = "trust_region/tests.rs"]
mod tests;
