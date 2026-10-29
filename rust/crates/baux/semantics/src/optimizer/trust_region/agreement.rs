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
}

impl Default for TrustRegionSpec {
    fn default() -> Self {
        Self {
            evidence_decay: 0.9,
            confidence_z: 2.0,
            reliability_threshold: 0.6,
            minimum_evidence: 4.0,
            acceptance_ratio: 0.1,
            boundary_fraction: 0.8,
            expand_factor: 2.0,
            shrink_factor: 0.5,
            failure_tolerance: 0,
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
    minimum: f64,
    maximum: f64,
    length: f64,
    adequate: f64,
    inadequate: f64,
    log_conversion: Option<f64>,
    telemetry: TrustRegionStatus,
    consecutive_failures: usize,
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
            consecutive_failures: 0,
        })
    }

    pub const fn length(&self) -> f64 {
        self.length
    }

    pub const fn telemetry(&self) -> TrustRegionStatus {
        self.telemetry
    }

    pub fn update(&mut self, evidence: TrialStep) -> Result<TrustRegionStatus, String> {
        validate_evidence(evidence)?;
        let observation_scale = evidence.observation_variance.sqrt();
        let prediction_scale = evidence.prediction_variance.sqrt();
        let scale = (evidence.observation_variance + evidence.prediction_variance)
            .sqrt()
            .max(MIN_SCALE);
        let error = (evidence.observed_improvement - evidence.predicted_improvement).abs() / scale;
        let response_scale = evidence
            .response_radius
            .max(observation_scale)
            .max(MIN_SCALE);
        let sharpness = prediction_scale / response_scale;
        let adequate = error <= self.config.confidence_z;
        self.update_adequacy(adequate);

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
        let model_adequate = ready && lower >= self.config.reliability_threshold;
        let model_inadequate = ready && upper < self.config.reliability_threshold;
        let ratio = (evidence.predicted_improvement > MIN_SCALE)
            .then_some(evidence.observed_improvement / evidence.predicted_improvement);
        let sharp = sharpness <= self.config.confidence_z;
        let boundary_step = reached.is_none_or(|value| value >= self.config.boundary_fraction);
        let failure = self.config.failure_tolerance > 0
            && !evidence.accepted
            && self.consecutive_failures.saturating_add(1) >= self.config.failure_tolerance;
        let action = decide_action(
            model_adequate,
            model_inadequate,
            sharp,
            evidence.accepted,
            ratio.is_some_and(|value| value >= self.config.acceptance_ratio),
            boundary_step,
            failure,
        );

        if evidence.accepted || action == RadiusAction::Expand {
            self.consecutive_failures = 0;
        } else if !evidence.accepted {
            if action == RadiusAction::Shrink {
                self.consecutive_failures = 0;
            } else {
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            }
        }

        self.length = match action {
            RadiusAction::Expand => self.length * self.config.expand_factor,
            RadiusAction::Shrink => self.length * self.config.shrink_factor,
            RadiusAction::Hold => self.length,
        }
        .clamp(self.minimum, self.maximum);
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
