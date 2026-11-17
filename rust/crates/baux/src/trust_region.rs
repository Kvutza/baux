//! Trust region implementations for TuRBO optimizer.

use ndarray::{Array1, ArrayView1, s};

mod null;
pub use null::{
    MorboTRSettings, NoTrustRegion, Rescalarize, TRLengthConfig, TrustRegionConfig,
    TrustRegionError, TrustRegionOutcome,
};

/// Trust region for single-objective optimization (TuRBO).
///
/// Implements success/failure-based length adaptation as a pure value state.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub const fn new(num_dim: usize, config: TRLengthConfig) -> Self {
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

    /// Pure consumed-`self` transition overriding the effective dimension.
    pub fn with_dim(mut self, dim: f64) -> Self {
        self.failure_tolerance_dim = Some(dim);
        self.with_computed()
    }

    /// Override the ambient dimension used for failure tolerance.
    pub fn set_dim(&mut self, dim: f64) {
        *self = self.with_dim(dim);
    }

    /// Pure consumed-`self` transition updating the active batch arm count.
    pub fn with_arms(mut self, num_arms: usize) -> Self {
        self.num_arms = Some(num_arms);
        self.with_computed()
    }

    /// Initialize or update batch size.
    pub fn set_arms(&mut self, num_arms: usize) {
        *self = self.with_arms(num_arms);
    }

    /// Pure consumed-`self` transition setting an explicit failure budget.
    pub fn with_tolerance(mut self, failures: usize) -> Result<Self, TrustRegionError> {
        let failures = i32::try_from(failures)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                TrustRegionError::InvalidParameter(
                    "failure tolerance must be in 1..=i32::MAX".into(),
                )
            })?;
        self.failure_tolerance_override = Some(failures);
        Ok(self.with_computed())
    }

    /// Use an explicit failure budget, independent of dimension and batch size.
    pub fn set_tolerance(&mut self, failures: usize) -> Result<(), TrustRegionError> {
        *self = self.with_tolerance(failures)?;
        Ok(())
    }

    /// Pure consumed-`self` transition toggling unit-hypercube clamping.
    pub const fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp_domain = clamp;
        self
    }

    /// Pure recomputation of failure tolerance from num_arms and num_dim.
    fn with_computed(mut self) -> Self {
        if let Some(failures) = self.failure_tolerance_override {
            self.failure_tolerance = Some(failures);
            return self;
        }
        if let Some(num_arms) = self.num_arms {
            let eff_dim = self.failure_tolerance_dim.unwrap_or(self.num_dim as f64);
            self.failure_tolerance = Some(baux::semantics::optimizer::turbo_tolerance(
                eff_dim, num_arms,
            ));
        }
        self
    }

    fn adapted(mut self, improved: Option<bool>) -> Self {
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
        self
    }

    /// Get current trust region length.
    #[inline(always)]
    pub const fn length(&self) -> f64 {
        self.length
    }

    /// Consecutive improvements required before the region expands.
    #[inline(always)]
    pub const fn success_tolerance(&self) -> i32 {
        self.success_tolerance
    }

    /// Consecutive non-improvements required before the region contracts.
    #[inline(always)]
    pub fn failure_tolerance(&self) -> i32 {
        self.failure_tolerance.unwrap_or(4)
    }

    /// Current consecutive-improvement count.
    #[inline(always)]
    pub const fn success_counter(&self) -> i32 {
        self.success_counter
    }

    /// Current consecutive-non-improvement count.
    #[inline(always)]
    pub const fn failure_counter(&self) -> i32 {
        self.failure_counter
    }

    /// Observation count at last trust-region update.
    #[inline(always)]
    pub const fn prev_obs(&self) -> usize {
        self.prev_obs
    }

    /// Pure consumed-`self` transition setting the incumbent scalar baseline.
    #[inline(always)]
    pub const fn with_best(mut self, value: f64) -> Self {
        self.best_value = value;
        self
    }

    /// Morbo: set incumbent scalar before an update step.
    pub fn set_best(&mut self, value: f64) {
        *self = self.with_best(value);
    }

    pub(crate) fn freeze_turbo(&self, buf: &mut Vec<u8>) {
        use crate::snapshot_io::{write_bool, write_int, write_len, write_real};
        write_real(buf, self.length);
        write_int(buf, self.failure_counter);
        write_int(buf, self.success_counter);
        write_real(buf, self.best_value);
        write_len(buf, self.prev_obs);
        write_int(buf, self.success_tolerance);
        match self.failure_tolerance {
            Some(ft) => {
                write_bool(buf, true);
                write_int(buf, ft);
            }
            None => write_bool(buf, false),
        }
        match self.num_arms {
            Some(na) => {
                write_bool(buf, true);
                write_len(buf, na);
            }
            None => write_bool(buf, false),
        }
        match self.failure_tolerance_dim {
            Some(fd) => {
                write_bool(buf, true);
                write_real(buf, fd);
            }
            None => write_bool(buf, false),
        }
        match self.failure_tolerance_override {
            Some(fo) => {
                write_bool(buf, true);
                write_int(buf, fo);
            }
            None => write_bool(buf, false),
        }
        write_bool(buf, self.clamp_domain);
    }

    pub(crate) fn thaw_turbo(
        &mut self,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<(), crate::error::ENNError> {
        use crate::snapshot_io::{MAX_OBS, bad_state};
        let length = reader.read_finite()?;
        if length <= 0.0 || length > self.config.length_max {
            return Err(bad_state("frozen TuRBO length out of valid range"));
        }
        let failure_counter = reader.read_int()?;
        let success_counter = reader.read_int()?;
        if failure_counter < 0 || success_counter < 0 {
            return Err(bad_state("frozen TuRBO counters must be non-negative"));
        }
        let best_value = reader.read_real()?;
        if best_value.is_nan() || best_value == f64::INFINITY {
            return Err(bad_state("invalid frozen TuRBO best_value"));
        }
        let prev_obs = reader.read_len(MAX_OBS)?;
        let success_tolerance = reader.read_int()?;
        if success_tolerance <= 0 {
            return Err(bad_state("frozen TuRBO success_tolerance must be > 0"));
        }
        let failure_tolerance = if reader.read_bool()? {
            let ft = reader.read_int()?;
            if ft <= 0 {
                return Err(bad_state("frozen TuRBO failure_tolerance must be > 0"));
            }
            Some(ft)
        } else {
            None
        };
        let num_arms = if reader.read_bool()? {
            let na = reader.read_len(MAX_OBS)?;
            if na == 0 {
                return Err(bad_state("frozen TuRBO num_arms must be > 0"));
            }
            Some(na)
        } else {
            None
        };
        let failure_tolerance_dim = if reader.read_bool()? {
            let fd = reader.read_finite()?;
            if fd <= 0.0 {
                return Err(bad_state("frozen TuRBO failure_tolerance_dim must be > 0"));
            }
            Some(fd)
        } else {
            None
        };
        let failure_tolerance_override = if reader.read_bool()? {
            let fo = reader.read_int()?;
            if fo <= 0 {
                return Err(bad_state(
                    "frozen TuRBO failure_tolerance_override must be > 0",
                ));
            }
            Some(fo)
        } else {
            None
        };
        let clamp_domain = reader.read_bool()?;
        self.length = length;
        self.failure_counter = failure_counter;
        self.success_counter = success_counter;
        self.best_value = best_value;
        self.prev_obs = prev_obs;
        self.success_tolerance = success_tolerance;
        self.failure_tolerance = failure_tolerance;
        self.num_arms = num_arms;
        self.failure_tolerance_dim = failure_tolerance_dim;
        self.failure_tolerance_override = failure_tolerance_override;
        self.clamp_domain = clamp_domain;
        Ok(())
    }

    pub(crate) fn hash_turbo(&self, buf: &mut Vec<u8>) {
        self.freeze_turbo(buf);
    }

    /// Pure consumed-`self` scalar incumbent step (MORBO / Python parity).
    pub fn step_scalar(
        mut self,
        num_obs: usize,
        y_incumbent_value: f64,
        scale: f64,
    ) -> Result<Self, TrustRegionError> {
        if num_obs == 0 || num_obs == self.prev_obs {
            return Ok(self);
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
            return Ok(self);
        }
        let improved = y_incumbent_value > self.best_value + 1e-3 * scale;
        self = self.adapted(Some(improved));
        self.best_value = self.best_value.max(y_incumbent_value);
        self.prev_obs = num_obs;
        Ok(self)
    }

    /// Morbo: update using zero-padded history and a scalar incumbent value
    /// (Python parity).
    pub fn update_scalar(
        &mut self,
        num_obs: usize,
        y_incumbent_value: f64,
        scale: f64,
    ) -> Result<(), TrustRegionError> {
        *self = self.step_scalar(num_obs, y_incumbent_value, scale)?;
        Ok(())
    }

    /// Check if restart is needed (length below minimum).
    #[inline(always)]
    pub fn needs_restart(&self) -> bool {
        self.length < self.config.length_min
    }

    /// Pure consumed-`self` update over full observation history.
    pub fn step_all(
        mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_obs {
            return Ok(self);
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

        if !self.best_value.is_finite() {
            let new_best = y_all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.best_value = new_best;
            self.prev_obs = n;
            return Ok(self);
        }

        let prev_slice = y_all.slice(s![..self.prev_obs]);
        let new_batch = y_all.slice(s![self.prev_obs..]);
        let new_best = new_batch.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let prev_len = prev_slice.len();
        let scale = if prev_len >= 2 {
            let min_val = prev_slice.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = prev_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (max_val - min_val).max(1e-6)
        } else {
            0.0
        };

        let improvement_threshold = 1e-3 * scale;
        let improved = new_best > self.best_value + improvement_threshold;
        if improved {
            self.best_value = self.best_value.max(new_best);
        }
        self = self.adapted(Some(improved));
        self.prev_obs = n;
        Ok(self)
    }

    /// Update trust region based on new observations.
    pub fn update(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), TrustRegionError> {
        *self = self.step_all(y_all, num_obs)?;
        Ok(())
    }

    /// Pure consumed-`self` update using observed history and incumbent scalar.
    pub fn step_history(
        self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<Self, TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_obs {
            return Ok(self);
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
        self.step_batch(&y_new, num_obs, y_incumbent_value)
    }

    /// Update trust region using observed history for scale and incumbent
    /// scalar for improvement.
    pub fn update_history(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        *self = self.step_history(y_all, num_obs, y_incumbent_value)?;
        Ok(())
    }

    /// Pure consumed-`self` O(batch) update over the newly told batch.
    pub fn step_batch(
        self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<Self, TrustRegionError> {
        step_sense(
            self,
            y_new,
            num_obs,
            y_incumbent_value,
            crate::sense::Maximize,
        )
    }

    /// Same as [`Self::update_history`], but only the newly told batch is
    /// required.
    pub fn update_batch(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        *self = self.step_batch(y_new, num_obs, y_incumbent_value)?;
        Ok(())
    }

    /// Pure consumed-`self` update from an explicit model-agreement outcome.
    pub fn step_outcome(
        mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
        outcome: TrustRegionOutcome,
    ) -> Result<Self, TrustRegionError> {
        if y_new.is_empty() {
            return Ok(self);
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
            return Ok(self);
        }

        let improved = match outcome {
            TrustRegionOutcome::Success => Some(true),
            TrustRegionOutcome::Failure => Some(false),
        };
        self = self.adapted(improved);

        if outcome == TrustRegionOutcome::Success {
            self.best_value = self.best_value.max(y_incumbent_value);
        }
        self.prev_obs = n;
        Ok(self)
    }

    /// Update from an explicit model-agreement outcome.
    pub fn update_outcome(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
        outcome: TrustRegionOutcome,
    ) -> Result<(), TrustRegionError> {
        *self = self.step_outcome(y_new, num_obs, y_incumbent_value, outcome)?;
        Ok(())
    }

    /// Zero-allocation scalar `(lower, upper)` bound query for one coordinate.
    #[inline(always)]
    pub fn bound_dim(&self, center: f64, scale: Option<f64>) -> (f64, f64) {
        let half = 0.5 * self.length * scale.unwrap_or(1.0);
        let lo = center - half;
        let hi = center + half;
        if self.clamp_domain {
            (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0))
        } else {
            (lo, hi)
        }
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
        let mut lb = Array1::zeros(num_dim);
        let mut ub = Array1::zeros(num_dim);
        for d in 0..num_dim {
            let scale = lengthscales.map(|ls| ls[d]);
            let (lo, hi) = self.bound_dim(x_center[d], scale);
            lb[d] = lo;
            ub[d] = hi;
        }
        (lb, ub)
    }

    /// Pure consumed-`self` restart preserving `prev_obs` watermark.
    pub const fn restarted(mut self) -> Self {
        self.length = self.config.length_init;
        self.failure_counter = 0;
        self.success_counter = 0;
        self.best_value = f64::NEG_INFINITY;
        self
    }

    /// Restart the trust region (reset to initial state).
    pub fn restart(&mut self) {
        *self = self.restarted();
    }

    /// Pure consumed-`self` complete local-region reset (`prev_obs = 0`).
    pub const fn restarted_local(mut self) -> Self {
        self = self.restarted();
        self.prev_obs = 0;
        self
    }

    /// Reset the complete local-region baseline after discarding its dataset.
    pub fn restart_local(&mut self) {
        *self = self.restarted_local();
    }

    /// Pure consumed-`self` watermark update without scanning history.
    pub const fn with_watermark(mut self, prev_obs: usize) -> Self {
        self.prev_obs = prev_obs;
        self
    }

    /// Set observation watermark without scanning history (post-init /
    /// restart).
    pub fn set_watermark(&mut self, prev_obs: usize) {
        *self = self.with_watermark(prev_obs);
    }

    /// Pure consumed-`self` baseline best-value and watermark initialization.
    pub const fn with_baseline(mut self, best_value: f64, prev_obs: usize) -> Self {
        if best_value.is_finite() {
            self.best_value = best_value;
        }
        self.prev_obs = prev_obs;
        self
    }

    /// Seed baseline best value and observation watermark (e.g. after warmup).
    pub fn seed_baseline(&mut self, best_value: f64, prev_obs: usize) {
        *self = self.with_baseline(best_value, prev_obs);
    }
}

/// Pure consumed-`self` sense-parametric trust-region batch update supporting
/// [`crate::sense::Maximize`], [`crate::sense::Minimize`], and
/// [`crate::sense::Target`].
pub fn step_sense<O: crate::sense::ObjectiveSense>(
    mut tr: TurboTrustRegion,
    y_new: &ArrayView1<f64>,
    num_obs: usize,
    y_incumbent_value: f64,
    sense: O,
) -> Result<TurboTrustRegion, TrustRegionError> {
    let n_new = y_new.len();
    if n_new == 0 {
        return Ok(tr);
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
        return Ok(tr);
    }

    let scale = improvement_scale(tr.best_value);
    let improved = utility > tr.best_value + 1e-3 * scale;
    tr = tr.adapted(Some(improved));
    tr.best_value = tr.best_value.max(utility);
    tr.prev_obs = n;
    Ok(tr)
}

/// Sense-parametric trust-region batch update adapter delegating to
/// [`step_sense`].
pub fn update_sense<O: crate::sense::ObjectiveSense>(
    tr: &mut TurboTrustRegion,
    y_new: &ArrayView1<f64>,
    num_obs: usize,
    y_incumbent_value: f64,
    sense: O,
) -> Result<(), TrustRegionError> {
    *tr = step_sense(*tr, y_new, num_obs, y_incumbent_value, sense)?;
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
