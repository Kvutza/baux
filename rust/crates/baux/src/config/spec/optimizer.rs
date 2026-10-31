//! Scientific roles in the TOML optimizer recipe, not executable algorithms.
//!
//! Proposal generation constructs trials; the surrogate estimates their values;
//! acquisition selects a trial; acceptance decides whether it replaces the
//! incumbent; radius control sets subsequent step sizes. Fitting parameters is
//! distinct from adapting a query's neighbor count. Definitions here describe
//! the configuration; `conversion` and runtime guards establish supported
//! paths. Deserialization success does not certify a theorem or learning
//! performance.

use super::*;

/// A closed optimizer recipe. The selected algorithm determines which
/// scientific roles exist; unrelated sections are rejected by deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "algorithm",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum OptimizerSpec {
    #[deser(rename = "operator-bo")]
    OperatorBo {
        proposal: crate::procedural_pool::ProposalMethod,
        #[deser(default)]
        distribution: crate::ProposalKernel,
        #[deser(default)]
        surrogate: SurrogateRecipe,
        #[deser(default)]
        trust_region: TrustRegionBounds,
    },
    #[deser(rename = "turbo-enn")]
    TurboEnn {
        proposal: crate::procedural_pool::ProposalMethod,
        #[deser(default = turbo_candidates())]
        candidates: u32,
        #[deser(default = default_arms())]
        arms: u32,
        distribution: Option<crate::ProposalKernel>,
        #[deser(flatten)]
        acquisition: AcquisitionSpec,
        #[deser(default)]
        surrogate: SurrogateRecipe,
        #[deser(default)]
        trust_region: TrustRegionBounds,
    },
    #[deser(rename = "muon")]
    Muon,
    #[deser(rename = "adamw")]
    Adamw,
}

impl OptimizerSpec {
    pub fn candidate_generation(&self) -> Option<CandidateGenerationSpec> {
        match self {
            Self::OperatorBo {
                proposal,
                distribution,
                ..
            } => Some(CandidateGenerationSpec {
                proposal: *proposal,
                candidates: 1,
                arms: 1,
                distribution: Some(*distribution),
            }),
            Self::TurboEnn {
                proposal,
                candidates,
                arms,
                distribution,
                ..
            } => Some(CandidateGenerationSpec {
                proposal: *proposal,
                candidates: *candidates,
                arms: *arms,
                distribution: *distribution,
            }),
            Self::Muon | Self::Adamw => None,
        }
    }
}

const fn default_arms() -> u32 {
    1
}

const fn turbo_candidates() -> u32 {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
/// Pool and transformation used to construct trials around an incumbent.
/// This does not set the number of forward evaluations or committed updates.
pub struct CandidateGenerationSpec {
    /// Transformation program, separate from the underlying noise distribution.
    pub proposal: crate::procedural_pool::ProposalMethod,
    /// TOTAL proposals across arms, distinct from fitting trials or objective
    /// calls.
    #[deser(default = default_arms())]
    pub candidates: u32,
    /// Seed namespaces sharing a center, not independent trust regions.
    #[deser(default = default_arms())]
    pub arms: u32,
    /// Underlying sampler selection; program transformations can change the
    /// distribution of the resulting parameter displacement.
    pub distribution: Option<crate::ProposalKernel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Geometry, uncertainty scales, and fitting policy for retained data.
/// These settings do not by themselves define how a trial direction is
/// generated.
pub struct SurrogateRecipe {
    /// Fixed k, or the configured neighbor budget when a local rule is enabled.
    pub neighbors: i32,
    /// Query-local neighbor policy, independent of hyperparameter fitting.
    pub neighbor_rule: NeighborRule,
    pub epistemic_scale: f64,
    pub aleatoric_scale: f64,
    pub y_scale: f64,
    /// How historical parameter dissimilarities are represented/computed;
    /// `latent` is not a reduced-dimensional parameterization of the model.
    pub geometry: HistoryGeometry,
    pub scaling: DistanceScaling,
    pub local_neighbors: Option<usize>,
    /// Scale fitting rule. This is independent of query-local neighbor choice.
    pub fit: SurrogateFit,
    /// Hyperparameter trials in the fitter, not candidate parameter vectors.
    pub fit_candidates: Option<usize>,
    /// Fitter's leave-one-out scoring budget, not fresh model evaluations.
    pub fit_samples: Option<usize>,
}

impl Default for SurrogateRecipe {
    fn default() -> Self {
        Self {
            neighbors: default_neighbors(),
            neighbor_rule: NeighborRule::default(),
            epistemic_scale: epistemic_default(),
            aleatoric_scale: aleatoric_default(),
            y_scale: default_scale(),
            geometry: HistoryGeometry::default(),
            scaling: DistanceScaling::default(),
            local_neighbors: None,
            fit: SurrogateFit::default(),
            fit_candidates: None,
            fit_samples: None,
        }
    }
}

impl SurrogateRecipe {
    pub fn fit_budget(&self) -> Result<(usize, usize), String> {
        match (self.fit, self.fit_candidates, self.fit_samples) {
            (SurrogateFit::Fixed, None, None) => Ok((0, 0)),
            (SurrogateFit::Fixed, _, _) => {
                Err("fit='fixed' cannot specify fit-candidates or fit-samples".into())
            }
            (SurrogateFit::LeaveOneOut, Some(candidates), Some(samples))
                if candidates > 0 && samples > 0 =>
            {
                Ok((candidates, samples))
            }
            (SurrogateFit::LeaveOneOut, _, _) => Err(
                "fit='leave-one-out' requires positive fit-candidates and fit-samples".into(),
            ),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum NeighborRule {
    /// Use the configured neighbor count, clamped to available history at ask.
    #[default]
    Fixed,
    /// Request the selection kernel's query-local risk-prefix rule.
    RiskPrefix,
}

/// Keep schema defaults aligned with the resident ask's initial neighbor
/// budget.
pub(super) fn default_neighbors() -> i32 {
    crate::trials::RoundConfig::default().surrogate.neighbors as i32
}

/// Initial epistemic scale from the same ask defaults used by the runtime.
pub(super) fn epistemic_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .epistemic_scale,
    )
}

/// Initial aleatoric scale from the resident ask, before optional fitting.
pub(super) fn aleatoric_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .aleatoric_scale,
    )
}

/// Initial output scale from the resident ask, before optional fitting.
pub(super) fn default_scale() -> f64 {
    f64::from(crate::trials::RoundConfig::default().surrogate.output_scale)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum SurrogateFit {
    /// Keep supplied hyperparameters; neighbor adaptation remains independent.
    #[default]
    Fixed,
    #[deser(rename = "leave-one-out")]
    LeaveOneOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "acquisition",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum AcquisitionSpec {
    #[deser(rename = "upper-confidence-bound")]
    Ucb {
        #[deser(default = default_beta())]
        beta: f64,
    },
    #[deser(rename = "thompson")]
    Thompson,
    /// Uniform selection control over the generated candidate pool.
    Random,
    Pareto {
        scales: Vec<f32>,
    },
}

#[path = "bounds.rs"]
mod bounds;
pub use bounds::*;

/// Reuse the shared UCB exploration default rather than a second local
/// constant.
fn default_beta() -> f64 {
    let AcquisitionConfig::UCB { beta } = AcquisitionConfig::default() else {
        unreachable!()
    };
    beta
}
