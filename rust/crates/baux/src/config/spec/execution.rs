//! Workload, data, randomness, diagnostics, and objective fields of a run.
//!
//! These settings complement the optimizer recipe; they are not interchangeable
//! measures of progress. Rounds count optimizer transactions, scored tokens
//! count objective work, and validation describes held-out quality. A latency
//! target is a requested measurement threshold, not a scheduling guarantee.

use super::*;

pub fn default_rounds() -> u32 {
    RunConfig::default().rounds()
}

pub fn default_reps() -> u32 {
    RunConfig::default().reps()
}

pub fn default_target() -> u32 {
    RunConfig::default().target_ms()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObjectiveFunction {
    #[default]
    CrossEntropy,
    BoundedLogLikelihood,
}

impl ObjectiveFunction {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CrossEntropy => "cross-entropy",
            Self::BoundedLogLikelihood => "bounded-log-likelihood",
        }
    }

    pub const fn score_mode(self) -> u32 {
        match self {
            Self::CrossEntropy => 1,
            Self::BoundedLogLikelihood => 2,
        }
    }

    /// Transform a per-token negative log-likelihood $z = -\log \pi_\theta(y_t
    /// \mid x_{<t})$ into the configured scalar token loss:
    /// - `CrossEntropy`: $z$
    /// - `BoundedLogLikelihood`: $\log(2 - \exp(-z)) =
    ///   \operatorname{ln\_1p}(-\operatorname{exp\_m1}(-z))$
    #[inline]
    pub fn token_loss(self, nll: f32) -> f32 {
        match self {
            Self::CrossEntropy => nll,
            Self::BoundedLogLikelihood => {
                if !nll.is_finite() {
                    return nll;
                }
                let z = f64::from(nll.max(0.0));
                ((-(-z).exp_m1()).ln_1p()) as f32
            }
        }
    }
}
