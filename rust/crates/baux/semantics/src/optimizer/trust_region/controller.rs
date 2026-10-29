//! Calibrated step acceptance and trust-region radius controller.

#[cfg(not(cabi_parity))]
use baux::semantics::calibration::Calibration;

pub(super) const MIN_SCALE: f64 = 1.0e-12;
pub(super) const PRIOR: f64 = 1.0;
const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
const FRAC_MASK: u64 = 0x000f_ffff_ffff_ffff;

pub type ReliabilityAction = RadiusAction;
pub type ReliabilityDecision = StepDecision;
#[cfg(not(cabi_parity))]
pub type RadiusController = TrustRegionController;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiusAction {
    Hold,
    Expand,
    Shrink,
}

/// Accept a measured improvement only when it strictly clears its error margin.
///
/// This is the executable scalar commit boundary shared by the optimizer
/// backends.  The Mathlib-facing specification is
/// `BauxResearch.Optimization.EnnCommit.accepts`.
pub fn accept_candidate(improvement: f64, margin: f64) -> bool {
    accept_bits(improvement.to_bits(), margin.to_bits())
}

fn nan_bits(bits: u64) -> bool {
    bits & EXP_MASK == EXP_MASK && bits & FRAC_MASK != 0
}

fn zero_bits(bits: u64) -> bool {
    bits & !SIGN_MASK == 0
}

fn order_bits(bits: u64) -> u64 {
    if bits & SIGN_MASK == 0 {
        bits | SIGN_MASK
    } else {
        !bits
    }
}

/// IEEE-754 binary64 strict comparison expressed only with integer operations.
///
/// NaNs are unordered and the two signed zeros compare equal, matching Rust's
/// native `f64` `>` relation.  Keeping the executable boundary in this form
/// lets Charon and Aeneas extract the actual comparison into Lean rather than
/// accepting a Boolean computed outside the checked model.
pub fn accept_bits(improvement: u64, margin: u64) -> bool {
    if nan_bits(improvement) || nan_bits(margin) {
        return false;
    }
    if zero_bits(improvement) && zero_bits(margin) {
        return false;
    }
    accept_decision(order_bits(improvement) > order_bits(margin))
}

/// Commit the candidate exactly when the measured gain clears its margin.
/// `accept_bits` performs the extracted binary64 comparison; this identity is
/// the final Boolean commit gate shared with the controller proof.
pub fn accept_decision(clears_margin: bool) -> bool {
    clears_margin
}

pub type Gate = bool;

/// Select the trust-region radius adaptation action from the controller's
/// Boolean gates.
///
/// This function is deliberately independent of floating-point arithmetic and
/// controller storage. Both production controllers call it after computing
/// their evidence predicates, and Aeneas translates this function to Lean.
pub fn decide_action(
    model_adequate: Gate,
    model_inadequate: Gate,
    sharp: Gate,
    accepted: Gate,
    positive_ratio: Gate,
    boundary_step: Gate,
    failure: Gate,
) -> RadiusAction {
    if model_inadequate {
        RadiusAction::Shrink
    } else if model_adequate && sharp && accepted && positive_ratio && boundary_step {
        RadiusAction::Expand
    } else if failure && !accepted {
        RadiusAction::Shrink
    } else {
        RadiusAction::Hold
    }
}

impl RadiusAction {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Expand => "expand",
            Self::Shrink => "shrink",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepDecision {
    pub numerical: baux::semantics::numerical::Outcome,
    pub accepted: bool,
    pub improvement: f64,
    pub threshold: f64,
    pub predicted_improvement: f64,
    pub action: RadiusAction,
    pub next_radius: f32,
    pub standardized_error: f64,
    pub sharpness_ratio: f64,
    pub agreement_ratio: Option<f64>,
}

#[cfg(not(cabi_parity))]
#[derive(Clone, Copy, Debug)]
struct TurboState {
    length: f64,
    minimum: f64,
    maximum: f64,
    success_limit: usize,
    failure_limit: usize,
    successes: usize,
    failures: usize,
}

#[cfg(not(cabi_parity))]
impl TurboState {
    fn step(&mut self, accepted: bool, fallback: f64) -> RadiusAction {
        if accepted {
            self.failures = 0;
            self.successes += 1;
            if self.successes >= self.success_limit {
                self.successes = 0;
                self.length = (self.length * 2.0).min(self.maximum);
                RadiusAction::Expand
            } else {
                RadiusAction::Hold
            }
        } else {
            self.failures += 1;
            self.successes = 0;
            if self.failures >= self.failure_limit {
                self.failures = 0;
                let shrunk = self.length * 0.5;
                if shrunk < self.minimum {
                    self.length = fallback;
                    self.successes = 0;
                } else {
                    self.length = shrunk;
                }
                RadiusAction::Shrink
            } else {
                RadiusAction::Hold
            }
        }
    }
}

/// Calibration and scalar step acceptance around the stochastic trust-region
/// controller. Invalid evidence leaves both states intact.
#[cfg(not(cabi_parity))]
pub struct TrustRegionController {
    pub(super) calibration: Calibration,
    control: super::agreement::ModelAgreement,
    turbo: Option<TurboState>,
}

#[cfg(not(cabi_parity))]
impl TrustRegionController {
    pub fn new(initial: f32, minimum: f32, maximum: f32) -> Result<Self, String> {
        Self::configured(initial, minimum, maximum, Default::default())
    }

    pub fn configured(
        initial: f32,
        minimum: f32,
        maximum: f32,
        policy: super::agreement::TrustRegionSpec,
    ) -> Result<Self, String> {
        Ok(Self {
            calibration: Calibration::default(),
            control: super::agreement::ModelAgreement::new(
                policy,
                f64::from(initial),
                f64::from(minimum),
                f64::from(maximum),
            )?,
            turbo: None,
        })
    }

    pub fn with_turbo(
        initial: f32,
        minimum: f32,
        maximum: f32,
        failure_limit: usize,
    ) -> Result<Self, String> {
        let control = super::agreement::ModelAgreement::new(
            Default::default(),
            f64::from(initial),
            f64::from(minimum),
            f64::from(maximum),
        )?;
        Ok(Self {
            calibration: Calibration::default(),
            control,
            turbo: Some(TurboState {
                length: f64::from(initial),
                minimum: f64::from(minimum),
                maximum: f64::from(maximum),
                success_limit: 3,
                failure_limit: failure_limit.max(1),
                successes: 0,
                failures: 0,
            }),
        })
    }

    pub fn from_policy(
        initial: f32,
        minimum: f32,
        maximum: f32,
        failure_limit: usize,
        policy: Option<super::agreement::TrustRegionSpec>,
    ) -> Result<Self, String> {
        Self::policy_f64(
            f64::from(initial),
            f64::from(minimum),
            f64::from(maximum),
            failure_limit,
            policy,
        )
    }

    pub fn policy_f64(
        initial: f64,
        minimum: f64,
        maximum: f64,
        failure_limit: usize,
        policy: Option<super::agreement::TrustRegionSpec>,
    ) -> Result<Self, String> {
        let control = super::agreement::ModelAgreement::new(
            policy.unwrap_or_default(),
            initial,
            minimum,
            maximum,
        )?;
        Ok(Self {
            calibration: Calibration::default(),
            control,
            turbo: policy.is_none().then_some(TurboState {
                length: initial,
                minimum,
                maximum,
                success_limit: 3,
                failure_limit: failure_limit.max(1),
                successes: 0,
                failures: 0,
            }),
        })
    }

    pub fn radius_f64(&self) -> f64 {
        self.turbo
            .map_or(self.control.length(), |state| state.length)
    }

    pub fn radius(&self) -> f32 {
        self.radius_f64() as f32
    }

    /// Warm observations use the supplied margin without advancing evidence or
    /// calibration. The incumbent radius remains unchanged.
    pub fn observe_initial(
        &self,
        observed_improvement: f32,
        predicted_improvement: f32,
        threshold: f32,
    ) -> Result<StepDecision, String> {
        self.initial_evidence(
            f64::from(observed_improvement),
            f64::from(predicted_improvement),
            f64::from(threshold),
        )
    }

    pub fn initial_evidence(
        &self,
        observed_improvement: f64,
        predicted_improvement: f64,
        threshold: f64,
    ) -> Result<StepDecision, String> {
        super::agreement::validate_evidence(super::agreement::TrialStep {
            observed_improvement,
            observation_variance: 0.0,
            predicted_improvement,
            prediction_variance: 0.0,
            accepted: false,
            nominal_radius: f64::from(self.radius()),
            response_radius: 0.0,
        })?;
        if !threshold.is_finite() {
            return Err("acceptance margin must be finite".into());
        }
        Ok(StepDecision {
            numerical: baux::semantics::numerical::unchecked("caller-supplied initial margin")?,
            accepted: accept_candidate(observed_improvement, threshold),
            improvement: observed_improvement,
            threshold,
            predicted_improvement,
            action: RadiusAction::Hold,
            next_radius: self.radius(),
            standardized_error: 0.0,
            sharpness_ratio: 0.0,
            agreement_ratio: (predicted_improvement > 0.0)
                .then_some(observed_improvement / predicted_improvement),
        })
    }

    /// Validate one measured observation, apply the common policy, then update
    /// calibration when requested. The acceptance margin uses only past errors.
    pub fn observe(
        &mut self,
        observed_improvement: f32,
        observation_variance: f32,
        predicted_improvement: f32,
        prediction_variance: f32,
        response_radius: f32,
        calibrate: bool,
    ) -> Result<StepDecision, String> {
        self.observe_evidence(
            super::agreement::TrialStep {
                observed_improvement: f64::from(observed_improvement),
                observation_variance: f64::from(observation_variance),
                predicted_improvement: f64::from(predicted_improvement),
                prediction_variance: f64::from(prediction_variance),
                accepted: false,
                nominal_radius: f64::from(self.radius()),
                response_radius: f64::from(response_radius),
            },
            calibrate,
        )
    }

    /// Self-contained round evaluation: computes observed and predicted
    /// improvements and combined observation/prediction variances internally,
    /// then runs the calibrated acceptance gate and radius controller update.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_step(
        &mut self,
        value: f32,
        variance: f32,
        incumbent_mean: f32,
        incumbent_se: f32,
        predicted_mean: f32,
        predicted_se: f32,
        radius: f32,
        calibrate: bool,
    ) -> Result<StepDecision, String> {
        let inc_se = f64::from(incumbent_se.max(0.0));
        let obs_var = f64::from(variance.max(0.0)) + inc_se * inc_se;
        let pred_se = f64::from(predicted_se.max(0.0));
        let r = f64::from(radius.abs());
        self.observe_evidence(
            super::agreement::TrialStep {
                observed_improvement: f64::from(value) - f64::from(incumbent_mean),
                observation_variance: obs_var,
                predicted_improvement: f64::from(predicted_mean) - f64::from(incumbent_mean),
                prediction_variance: pred_se * pred_se + inc_se * inc_se,
                accepted: false,
                nominal_radius: r,
                response_radius: r,
            },
            calibrate,
        )
    }

    /// Evaluate a trial step directly from the surrogate's `PoolSelection`,
    /// handling warmup vs. modeled baseline and calibration regime internally.
    pub fn evaluate_selection(
        &mut self,
        value: f32,
        variance: f32,
        radius: f32,
        selection: &baux::semantics::optimizer::surrogate::PoolSelection,
    ) -> Result<StepDecision, String> {
        if !variance.is_finite() || variance < 0.0 {
            return Err("observation variance must be finite and non-negative".into());
        }
        if selection.initializing {
            let improvement = f64::from(value) - f64::from(selection.baseline_value);
            let predicted =
                f64::from(selection.candidate.mean) - f64::from(selection.baseline_value);
            let base_var = selection.baseline_se * selection.baseline_se;
            let pair_var = if variance.to_bits() == base_var.to_bits() {
                0.0
            } else {
                f64::from(variance)
            };
            let multiplier = f64::from(self.calibration.multiplier());
            let threshold = multiplier * pair_var.sqrt();
            super::agreement::validate_evidence(super::agreement::TrialStep {
                observed_improvement: improvement,
                observation_variance: 0.0,
                predicted_improvement: predicted,
                prediction_variance: 0.0,
                accepted: false,
                nominal_radius: f64::from(self.radius()),
                response_radius: 0.0,
            })?;
            if !threshold.is_finite() {
                return Err("acceptance margin must be finite".into());
            }
            let accepted = accept_candidate(improvement, threshold);
            let numerical =
                baux::semantics::numerical::check(baux::semantics::numerical::Trace::observed(
                    baux::semantics::numerical::Kind::Paired,
                    improvement,
                    multiplier,
                    pair_var,
                    0.0,
                    threshold,
                    accepted,
                ))?;
            return Ok(StepDecision {
                numerical,
                accepted,
                improvement,
                threshold,
                predicted_improvement: predicted,
                action: RadiusAction::Hold,
                next_radius: self.radius(),
                standardized_error: 0.0,
                sharpness_ratio: 0.0,
                agreement_ratio: (predicted > 0.0).then_some(improvement / predicted),
            });
        }
        if let Some(mut turbo_state) = self.turbo {
            let improvement = f64::from(value) - f64::from(selection.baseline_value);
            let predicted =
                f64::from(selection.candidate.mean) - f64::from(selection.incumbent.mean);
            let pred_se = f64::from(selection.candidate.standard_error.max(0.0));
            let obs_var = f64::from(variance);
            let pred_var = pred_se * pred_se;
            let r = f64::from(radius.abs());
            super::agreement::validate_evidence(super::agreement::TrialStep {
                observed_improvement: improvement,
                observation_variance: obs_var,
                predicted_improvement: predicted,
                prediction_variance: pred_var,
                accepted: false,
                nominal_radius: f64::from(self.radius()),
                response_radius: r,
            })?;
            let threshold = 0.0;
            let accepted = accept_candidate(improvement, threshold);
            let numerical =
                baux::semantics::numerical::check(baux::semantics::numerical::Trace::observed(
                    baux::semantics::numerical::Kind::Paired,
                    improvement,
                    0.0,
                    obs_var,
                    0.0,
                    threshold,
                    accepted,
                ))?;
            let mut calibration = self.calibration.clone();
            calibration.observe(
                improvement as f32,
                predicted as f32,
                (obs_var + pred_var).sqrt() as f32,
            )?;
            let action = turbo_state.step(accepted, self.control.length());
            self.calibration = calibration;
            self.turbo = Some(turbo_state);
            return Ok(StepDecision {
                numerical,
                accepted,
                improvement,
                threshold,
                predicted_improvement: predicted,
                action,
                next_radius: self.radius(),
                standardized_error: 0.0,
                sharpness_ratio: 0.0,
                agreement_ratio: (predicted > 0.0).then_some(improvement / predicted),
            });
        }
        self.evaluate_step(
            value,
            variance,
            selection.baseline_value,
            selection.baseline_se,
            selection.candidate.mean,
            selection.candidate.standard_error,
            radius,
            true,
        )
    }

    /// Preserve the FP64 evidence arithmetic used by the Metal adapter. The
    /// acceptance gate is computed here; the caller's accepted flag is ignored.
    pub fn observe_evidence(
        &mut self,
        mut evidence: super::agreement::TrialStep,
        calibrate: bool,
    ) -> Result<StepDecision, String> {
        let improvement = evidence.observed_improvement;
        let multiplier = f64::from(self.calibration.multiplier());
        let threshold = multiplier * evidence.observation_variance.sqrt();
        let accepted = accept_candidate(improvement, threshold);
        let numerical =
            baux::semantics::numerical::check(baux::semantics::numerical::Trace::observed(
                baux::semantics::numerical::Kind::Paired,
                improvement,
                multiplier,
                evidence.observation_variance,
                0.0,
                threshold,
                accepted,
            ))?;
        let mut calibration = self.calibration.clone();
        if calibrate {
            calibration.observe(
                improvement as f32,
                evidence.predicted_improvement as f32,
                (evidence.observation_variance + evidence.prediction_variance).sqrt() as f32,
            )?;
        }
        evidence.accepted = accepted;
        let result = self.control.update(evidence)?;
        self.calibration = calibration;
        Ok(StepDecision {
            numerical,
            accepted,
            improvement,
            threshold,
            predicted_improvement: evidence.predicted_improvement,
            action: result.action,
            next_radius: self.radius(),
            standardized_error: result.standardized_error.unwrap(),
            sharpness_ratio: result.sharpness_ratio.unwrap(),
            agreement_ratio: result.agreement_ratio,
        })
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn prove_accept() {
        let left: u64 = kani::any();
        let right: u64 = kani::any();
        let lf = f64::from_bits(left);
        let rf = f64::from_bits(right);
        assert_eq!(accept_bits(left, right), lf > rf);
        assert!(!accept_bits(left, left));
    }

    #[kani::proof]
    fn prove_action() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let c: bool = kani::any();
        let d: bool = kani::any();
        let e: bool = kani::any();
        let f: bool = kani::any();
        let g: bool = kani::any();
        let action = decide_action(a, b, c, d, e, f, g);
        if b {
            assert_eq!(action, RadiusAction::Shrink);
        }
    }
}
