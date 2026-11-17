use deser::{Deserialize, Serialize};
use ndarray::{Array1, ArrayView1};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum Rescalarize {
    OnRestart,
    #[deser(alias = "every-step")]
    OnPropose,
}

impl Rescalarize {
    /// Alias matching multi-objective per-step rescalarization terminology.
    #[allow(non_upper_case_globals)]
    pub const EveryStep: Self = Self::OnPropose;

    /// Convert to `#![no_std]` [`baux::semantics::morbo::RescalarizeMode`].
    #[inline(always)]
    pub const fn to_mode(self) -> baux::semantics::morbo::RescalarizeMode {
        match self {
            Self::OnRestart => baux::semantics::morbo::RescalarizeMode::OnRestart,
            Self::OnPropose => baux::semantics::morbo::RescalarizeMode::OnPropose,
        }
    }

    /// Convert from `#![no_std]` [`baux::semantics::morbo::RescalarizeMode`].
    #[inline(always)]
    pub const fn from_mode(mode: baux::semantics::morbo::RescalarizeMode) -> Self {
        match mode {
            baux::semantics::morbo::RescalarizeMode::OnRestart => Self::OnRestart,
            baux::semantics::morbo::RescalarizeMode::OnPropose => Self::OnPropose,
        }
    }
}

impl std::str::FromStr for Rescalarize {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "on-restart" | "on_restart" => Ok(Self::OnRestart),
            "on-propose" | "on_propose" | "every-step" | "every_step" => Ok(Self::OnPropose),
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

/// Null trust region (no trust region management).
///
/// Always returns full bounds [0, 1]^d and never needs restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoTrustRegion {
    num_dim: usize,
}

impl NoTrustRegion {
    /// Create a new NoTrustRegion.
    pub const fn new(num_dim: usize) -> Self {
        Self { num_dim }
    }

    /// Ambient coordinate dimension.
    #[inline(always)]
    pub const fn num_dim(&self) -> usize {
        self.num_dim
    }

    /// Get current length (always 1.0).
    #[inline(always)]
    pub const fn length(&self) -> f64 {
        1.0
    }

    /// Check if restart is needed (always false).
    #[inline(always)]
    pub const fn needs_restart(&self) -> bool {
        false
    }

    /// Zero-allocation per-coordinate unit interval `[0.0, 1.0]`.
    #[inline(always)]
    pub const fn bound_dim(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Pure consumed-`self` observation update (identity transition).
    #[inline(always)]
    pub const fn updated(self, _y_new: &ArrayView1<f64>, _num_obs: usize) -> Self {
        self
    }

    /// Pure consumed-`self` restart transition (identity transition).
    #[inline(always)]
    pub const fn restarted(self) -> Self {
        self
    }

    /// Update (no-op adapter delegating to [`Self::updated`]).
    pub fn update(&mut self, y_new: &ArrayView1<f64>, num_obs: usize) {
        *self = self.updated(y_new, num_obs);
    }

    /// Compute trust region bounds (always returns [0, 1]^d).
    pub fn compute_bounds(
        &self,
        _x_center: &ArrayView1<f64>,
        _lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let lb = Array1::zeros(self.num_dim);
        let ub = Array1::ones(self.num_dim);
        (lb, ub)
    }

    /// Restart (no-op adapter delegating to [`Self::restarted`]).
    pub fn restart(&mut self) {
        *self = self.restarted();
    }
}
