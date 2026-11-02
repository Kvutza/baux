//! Stochastic trust-region control from surrogate model agreement.
//!
//! Acceptance and region adaptation are deliberately separate. Acceptance
//! asks whether the measured objective improved. Region adaptation asks
//! whether the surrogate predicted that measured change within its stated
//! uncertainty. A region is enlarged only after an accurate, useful boundary
//! step and is contracted when accumulated evidence says the local model is
//! inaccurate.

pub use super::controller::RadiusAction;
use super::controller::{MIN_SCALE, PRIOR, decide_action};
use deser::{Deserialize, Serialize};

pub type ReliabilityPolicy = TrustRegionSpec;
pub type ReliabilityEvidence = TrialStep;
pub type ReliabilityTelemetry = TrustRegionStatus;
pub type RadiusControl = ModelAgreement;

/// Specification for stochastic trust-region model-agreement control.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct TrustRegionSpec {
    /// Discount applied to past adequate/inadequate model evidence.
    pub evidence_decay: f64,
    /// Standardized residual tolerated from a locally adequate model.
    pub confidence_z: f64,
    /// Lower adequacy-evidence bound required for expansion.
    pub reliability_threshold: f64,
    /// Effective observations required before adequacy changes the radius.
    pub minimum_evidence: f64,
    /// Minimum realized-to-predicted improvement ratio for expansion.
    pub acceptance_ratio: f64,
    /// Minimum fraction of the estimated response boundary reached by a step.
    pub boundary_fraction: f64,
    /// Multiplicative expansion after an adequate accepted boundary step.
    pub expand_factor: f64,
    /// Multiplicative contraction when the local model is inadequate.
    pub shrink_factor: f64,
    /// Consecutive non-improvements before contracting the radius.
    /// Default is 0 (disabled unless explicitly configured).
    pub failure_tolerance: usize,
    /// Unconstrained continuous descent mode: always accept candidate steps.
    pub continuous_descent: bool,
}

impl Default for TrustRegionSpec {
    fn default() -> Self {
        Self {
            evidence_decay: 0.9,
            confidence_z: 2.0,
            reliability_threshold: 0.6,
            minimum_evidence: 4.0,
            acceptance_ratio: 0.1,
            boundary_fraction: 0.5,
            expand_factor: 2.0,
            shrink_factor: 0.5,
            failure_tolerance: 0,
            continuous_descent: false,
        }
    }
}

impl TrustRegionSpec {
    pub fn validate(self) -> Result<Self, String> {
        for (name, value) in [
            ("evidence_decay", self.evidence_decay),
            ("reliability_threshold", self.reliability_threshold),
            ("boundary_fraction", self.boundary_fraction),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("model-agreement {name} must be in 0..=1"));
            }
        }
        for (name, value) in [
            ("confidence_z", self.confidence_z),
            ("minimum_evidence", self.minimum_evidence),
            ("acceptance_ratio", self.acceptance_ratio),
            ("expand_factor", self.expand_factor),
            ("shrink_factor", self.shrink_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "model-agreement {name} must be positive and finite"
                ));
            }
        }
        if self.expand_factor <= 1.0 || self.shrink_factor >= 1.0 {
            return Err("model-agreement requires expand_factor > 1 and shrink_factor < 1".into());
        }
        Ok(self)
    }
}

/// Realized vs. predicted outcome from one black-box trial step evaluation.
#[derive(Debug, Clone, Copy)]
pub struct TrialStep {
    pub observed_improvement: f64,
    pub observation_variance: f64,
    pub predicted_improvement: f64,
    pub prediction_variance: f64,
    pub accepted: bool,
    /// Nominal proposal scale before quantization and operator response.
    pub nominal_radius: f64,
    /// Realized response-space displacement. mHC-4 supplies the RMS causal
    /// patch response; scalar objectives fall back to normalized weight
    /// displacement reduced by the proposal kernel.
    pub response_radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustRegionStatus {
    pub action: RadiusAction,
    pub standardized_error: Option<f64>,
    pub sharpness_ratio: Option<f64>,
    pub agreement_ratio: Option<f64>,
    pub adequacy_mean: f64,
    pub adequacy_lower: f64,
    pub evidence: f64,
    pub response_radius: Option<f64>,
    pub response_boundary: Option<f64>,
    pub boundary_fraction: Option<f64>,
    pub conversion: Option<f64>,
    pub length: f64,
}

/// Discounted Beta posterior controller over probabilistic surrogate model
/// agreement.
#[derive(Debug, Clone)]
pub struct ModelAgreement {
    config: TrustRegionSpec,
    initial: f64,
    minimum: f64,
    maximum: f64,
    length: f64,
    adequate: f64,
    inadequate: f64,
    log_conversion: Option<f64>,
    telemetry: TrustRegionStatus,
    consecutive_successes: usize,
    consecutive_failures: usize,
    floor_bounces: usize,
}

impl ModelAgreement {
    pub fn new(
        config: TrustRegionSpec,
        initial: f64,
        minimum: f64,
        maximum: f64,
    ) -> Result<Self, String> {
        let config = config.validate()?;
        if !initial.is_finite()
            || !minimum.is_finite()
            || !maximum.is_finite()
            || minimum <= 0.0
            || minimum > initial
            || initial > maximum
        {
            return Err(
                "trust-region radius bounds must be finite and satisfy 0 < min <= initial <= max"
                    .into(),
            );
        }
        let (mean, lower, _) = beta_summary(PRIOR, PRIOR, config.confidence_z);
        Ok(Self {
            config,
            initial,
            minimum,
            maximum,
            length: initial,
            adequate: PRIOR,
            inadequate: PRIOR,
            log_conversion: None,
            telemetry: TrustRegionStatus {
                action: RadiusAction::Hold,
                standardized_error: None,
                sharpness_ratio: None,
                agreement_ratio: None,
                adequacy_mean: mean,
                adequacy_lower: lower,
                evidence: 0.0,
                response_radius: None,
                response_boundary: None,
                boundary_fraction: None,
                conversion: None,
                length: initial,
            },
            consecutive_successes: 0,
            consecutive_failures: 0,
            floor_bounces: 0,
        })
    }

    pub const fn length(&self) -> f64 {
        self.length
    }

    pub const fn telemetry(&self) -> TrustRegionStatus {
        self.telemetry
    }

    pub const fn continuous_descent(&self) -> bool {
        self.config.continuous_descent
    }

    pub fn update(&mut self, evidence: TrialStep) -> Result<TrustRegionStatus, String> {
        validate_evidence(evidence)?;
        let scale = (evidence.observation_variance + evidence.prediction_variance)
            .sqrt()
            .max(MIN_SCALE);
        let error = (evidence.observed_improvement - evidence.predicted_improvement).abs() / scale;
        let sharpness = evidence.prediction_variance.sqrt()
            / evidence
                .response_radius
                .max(evidence.observation_variance.sqrt())
                .max(MIN_SCALE);
        self.update_adequacy(error <= self.config.confidence_z);

        let previous_conversion = self.log_conversion.map(f64::exp);
        let response_boundary = previous_conversion.map(|value| value * evidence.nominal_radius);
        let reached = response_boundary
            .filter(|value| *value > 0.0)
            .map(|value| evidence.response_radius / value);
        self.update_conversion(evidence.nominal_radius, evidence.response_radius);

        let (mean, lower, upper) =
            beta_summary(self.adequate, self.inadequate, self.config.confidence_z);
        let amount = (self.adequate + self.inadequate - 2.0 * PRIOR).max(0.0);
        let ready = amount >= self.config.minimum_evidence;
        let streak_expand = evidence.accepted && self.consecutive_successes.saturating_add(1) >= 3;
        let model_adequate = (ready && lower >= self.config.reliability_threshold) || streak_expand;
        let model_inadequate =
            ready && upper < self.config.reliability_threshold && !evidence.accepted;
        let ratio = (evidence.predicted_improvement > MIN_SCALE)
            .then_some(evidence.observed_improvement / evidence.predicted_improvement);
        let positive_ratio = ratio.map_or(evidence.observed_improvement > 0.0, |value| {
            value >= self.config.acceptance_ratio
        }) || streak_expand;
        let sharp = sharpness <= self.config.confidence_z || streak_expand;
        let boundary_step =
            reached.is_none_or(|value| value >= self.config.boundary_fraction) || streak_expand;
        let failure = !self.config.continuous_descent
            && self.config.failure_tolerance > 0
            && !evidence.accepted
            && self.consecutive_failures.saturating_add(1) >= self.config.failure_tolerance;
        let action = if self.config.continuous_descent {
            continuous_action(
                evidence.accepted,
                error,
                sharpness,
                self.config.confidence_z,
                self.consecutive_successes,
                self.consecutive_failures,
            )
        } else {
            decide_action(
                model_adequate,
                model_inadequate,
                sharp,
                evidence.accepted,
                positive_ratio,
                boundary_step,
                failure,
            )
        };

        self.apply_action(action, evidence.accepted);
        self.telemetry = TrustRegionStatus {
            action,
            standardized_error: Some(error),
            sharpness_ratio: Some(sharpness),
            agreement_ratio: ratio,
            adequacy_mean: mean,
            adequacy_lower: lower,
            evidence: amount,
            response_radius: (evidence.response_radius > 0.0).then_some(evidence.response_radius),
            response_boundary,
            boundary_fraction: reached,
            conversion: self.log_conversion.map(f64::exp),
            length: self.length,
        };
        Ok(self.telemetry)
    }

    fn apply_action(&mut self, action: RadiusAction, accepted: bool) {
        if accepted {
            self.consecutive_failures = 0;
            self.floor_bounces = 0;
            if action == RadiusAction::Expand {
                self.consecutive_successes = 0;
            } else {
                self.consecutive_successes = self.consecutive_successes.saturating_add(1);
            }
        } else {
            self.consecutive_successes = 0;
            if action == RadiusAction::Shrink {
                self.consecutive_failures = 0;
                self.adequate = PRIOR;
                self.inadequate = PRIOR;
            } else {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            }
        }
        self.length = match action {
            RadiusAction::Expand => {
                (self.length * self.config.expand_factor).clamp(self.minimum, self.maximum)
            }
            RadiusAction::Shrink => {
                let shrunk = self.length * self.config.shrink_factor;
                if shrunk >= self.minimum * (1.0 - 1.0e-5) {
                    shrunk.clamp(self.minimum, self.maximum)
                } else if self.config.continuous_descent {
                    if self.floor_bounces == 0 {
                        self.floor_bounces = 1;
                        (self.minimum / self.config.shrink_factor).clamp(self.minimum, self.initial)
                    } else {
                        self.minimum
                    }
                } else {
                    self.initial.clamp(self.minimum, self.maximum)
                }
            }
            RadiusAction::Hold => self.length.clamp(self.minimum, self.maximum),
        };
    }

    fn update_adequacy(&mut self, adequate: bool) {
        self.adequate = PRIOR + self.config.evidence_decay * (self.adequate - PRIOR);
        self.inadequate = PRIOR + self.config.evidence_decay * (self.inadequate - PRIOR);
        if adequate {
            self.adequate += 1.0;
        } else {
            self.inadequate += 1.0;
        }
    }

    fn update_conversion(&mut self, nominal: f64, response: f64) {
        if response <= 0.0 {
            return;
        }
        let sample = (response / nominal).ln();
        self.log_conversion = Some(self.log_conversion.map_or(sample, |previous| {
            self.config.evidence_decay * previous + (1.0 - self.config.evidence_decay) * sample
        }));
    }
}

fn beta_summary(success: f64, failure: f64, z: f64) -> (f64, f64, f64) {
    let total = success + failure;
    let mean = success / total;
    let variance = success * failure / (total * total * (total + 1.0));
    let half_width = z * variance.sqrt();
    (
        mean,
        (mean - half_width).clamp(0.0, 1.0),
        (mean + half_width).clamp(0.0, 1.0),
    )
}

pub(super) fn validate_evidence(evidence: TrialStep) -> Result<(), String> {
    let finite = [
        evidence.observed_improvement,
        evidence.observation_variance,
        evidence.predicted_improvement,
        evidence.prediction_variance,
        evidence.nominal_radius,
        evidence.response_radius,
    ];
    if finite.iter().any(|value| !value.is_finite())
        || evidence.observation_variance < 0.0
        || evidence.prediction_variance < 0.0
        || evidence.nominal_radius <= 0.0
        || evidence.response_radius < 0.0
    {
        return Err("invalid model-agreement evidence".into());
    }
    Ok(())
}

fn continuous_action(
    accepted: bool,
    error: f64,
    sharpness: f64,
    confidence_z: f64,
    consecutive_successes: usize,
    consecutive_failures: usize,
) -> RadiusAction {
    if accepted
        && (consecutive_successes >= 1 || (error <= 1.25 && sharpness <= confidence_z * 1.5))
    {
        RadiusAction::Expand
    } else if !accepted && consecutive_failures >= 2 {
        RadiusAction::Shrink
    } else {
        RadiusAction::Hold
    }
}
