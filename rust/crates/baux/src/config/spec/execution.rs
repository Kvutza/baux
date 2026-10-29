//! Workload, data, randomness, diagnostics, and objective fields of a run.
//!
//! These settings complement the optimizer recipe; they are not interchangeable
//! measures of progress. Rounds count optimizer transactions, scored tokens
//! count objective work, and validation describes held-out quality. A latency
//! target is a requested measurement threshold, not a scheduling guarantee.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Requested execution budget, separate from algorithm and evaluation policy.
pub struct ExecutionSpec {
    /// Optimizer transactions per repetition, not epochs or successful updates.
    pub rounds: u32,
    /// Requested repetitions; seed/config handling is resolved by the runner.
    pub repetitions: u32,
    /// Requested round-wall latency in milliseconds, not an achieved bound.
    pub round_target_ms: u32,
}

impl Default for ExecutionSpec {
    fn default() -> Self {
        let config = RunConfig::default();
        Self {
            rounds: config.rounds(),
            repetitions: config.reps(),
            round_target_ms: config.target_ms(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Held-out measurement policy. Validation observations never train the
/// optimizer.
pub struct EvaluationSpec {
    /// Validation cadence in optimizer transactions.
    pub interval: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct DataSpec {
    /// Versioned source/filter/tokenization/split recipe resolved by
    /// preparation.
    pub recipe: Option<DataRecipe>,
    pub train: Option<PathBuf>,
    pub validation: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Optional instrumentation. Enabling diagnostics can change measured overhead;
/// their output is evidence about that scope, not a mathematical certificate.
pub struct DiagnosticSpec {
    pub trace: bool,
    pub stage_samples: Option<u32>,
    pub pool_rank: Option<u32>,
    pub kernels: Option<KernelTrial>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Objective protocol requested by the fixed-corpus path.
/// The runtime must still define scored positions, sign, and estimate variance.
pub struct ObjectiveSpec {
    pub function: Option<ObjectiveFunction>,
    pub sampling: Option<ObjectiveSampling>,
    /// Requested corpus tokens per causal-likelihood window, including boundary
    /// context; actual scored positions are reported separately by the
    /// evaluator.
    pub window_tokens: Option<u32>,
    pub reference: Option<ObjectiveReference>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObjectiveSampling {
    Natural,
}
