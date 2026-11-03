//! Step acceptance and trust-region radius controller.

const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
const FRAC_MASK: u64 = 0x000f_ffff_ffff_ffff;

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
    pub agreement_ratio: Option<f64>,
}

#[cfg(not(cabi_parity))]
pub fn turbo_tolerance(num_dim: f64, num_arms: usize) -> i32 {
    let arms = (num_arms.max(1)) as f64;
    let tolerance = ((4.0 / arms).max(num_dim / arms)).ceil() as i32;
    tolerance.max(1)
}

#[cfg(not(cabi_parity))]
pub fn turbo_adapt(
    length: &mut f64,
    length_max: f64,
    success_counter: &mut i32,
    failure_counter: &mut i32,
    success_tol: i32,
    failure_tol: i32,
    improved: Option<bool>,
) -> RadiusAction {
    match improved {
        Some(true) => {
            *success_counter += 1;
            *failure_counter = 0;
        }
        Some(false) => {
            *failure_counter += 1;
            *success_counter = 0;
        }
        None => {}
    }
    if *success_counter >= success_tol {
        *length = (*length * 2.0).min(length_max);
        *success_counter = 0;
        RadiusAction::Expand
    } else if *failure_counter >= failure_tol {
        *length *= 0.5;
        *failure_counter = 0;
        RadiusAction::Shrink
    } else {
        RadiusAction::Hold
    }
}

#[cfg(not(cabi_parity))]
#[derive(Clone, Copy, Debug)]
pub struct TurboState {
    pub length: f64,
    pub initial: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub success_limit: usize,
    pub failure_limit: usize,
    pub successes: usize,
    pub failures: usize,
}

#[cfg(not(cabi_parity))]
impl TurboState {
    pub fn new(initial: f64, minimum: f64, maximum: f64, failure_limit: usize) -> Self {
        Self {
            length: initial,
            initial,
            minimum,
            maximum,
            success_limit: 3,
            failure_limit: failure_limit.max(1),
            successes: 0,
            failures: 0,
        }
    }

    pub fn step(&mut self, accepted: bool) -> RadiusAction {
        let mut successes = self.successes as i32;
        let mut failures = self.failures as i32;
        let action = turbo_adapt(
            &mut self.length,
            self.maximum,
            &mut successes,
            &mut failures,
            self.success_limit as i32,
            self.failure_limit as i32,
            Some(accepted),
        );
        self.successes = successes as usize;
        self.failures = failures as usize;
        if action == RadiusAction::Shrink && self.length < self.minimum {
            // TuRBO restart: collapse below minimum resets to initial.
            self.length = self.initial;
            self.successes = 0;
        }
        action
    }
}

/// Pure TuRBO trust-region adaptation with numerical validation.
#[cfg(not(cabi_parity))]
pub struct TrustRegionController {
    turbo: TurboState,
}

#[cfg(not(cabi_parity))]
pub(super) fn validate_improvements(observed: f64, predicted: f64) -> Result<(), String> {
    if !observed.is_finite() || !predicted.is_finite() {
        return Err("trial-step improvements must be finite".into());
    }
    Ok(())
}

#[cfg(not(cabi_parity))]
impl TrustRegionController {
    pub fn new(initial: f32, minimum: f32, maximum: f32) -> Result<Self, String> {
        Self::with_turbo(initial, minimum, maximum, 4)
    }

    pub fn with_turbo(
        initial: f32,
        minimum: f32,
        maximum: f32,
        failure_limit: usize,
    ) -> Result<Self, String> {
        Self::policy_f64(
            f64::from(initial),
            f64::from(minimum),
            f64::from(maximum),
            failure_limit,
        )
    }

    pub fn policy_f64(
        initial: f64,
        minimum: f64,
        maximum: f64,
        failure_limit: usize,
    ) -> Result<Self, String> {
        if !initial.is_finite() || !minimum.is_finite() || !maximum.is_finite() {
            return Err("trust-region boundaries must be finite".into());
        }
        if minimum <= 0.0 || initial < minimum || initial > maximum {
            return Err("invalid trust-region radius boundaries".into());
        }
        Ok(Self {
            turbo: TurboState::new(initial, minimum, maximum, failure_limit),
        })
    }

    pub fn radius_f64(&self) -> f64 {
        self.turbo.length
    }

    pub fn radius(&self) -> f32 {
        self.radius_f64() as f32
    }

    /// Warm observations use the supplied margin without advancing evidence.
    /// The incumbent radius remains unchanged.
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
        validate_improvements(observed_improvement, predicted_improvement)?;
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
            agreement_ratio: (predicted_improvement > 0.0)
                .then_some(observed_improvement / predicted_improvement),
        })
    }

    /// Evaluate a trial step directly from the surrogate's `PoolSelection`,
    /// handling warmup vs. modeled baseline internally.
    pub fn evaluate_selection(
        &mut self,
        value: f32,
        variance: f32,
        _radius: f32,
        selection: &baux::semantics::optimizer::surrogate::PoolSelection,
    ) -> Result<StepDecision, String> {
        if !variance.is_finite() || variance < 0.0 {
            return Err("observation variance must be finite and non-negative".into());
        }
        if selection.initializing {
            return eval_warmup(self, value, variance, selection);
        }
        let improvement = f64::from(value) - f64::from(selection.baseline_value);
        let predicted = f64::from(selection.candidate.mean) - f64::from(selection.incumbent.mean);
        let obs_var = f64::from(variance);
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
        let action = self.turbo.step(accepted);
        Ok(StepDecision {
            numerical,
            accepted,
            improvement,
            threshold,
            predicted_improvement: predicted,
            action,
            next_radius: self.radius(),
            agreement_ratio: (predicted > 0.0).then_some(improvement / predicted),
        })
    }
}

#[cfg(not(cabi_parity))]
fn eval_warmup(
    ctrl: &RadiusController,
    value: f32,
    variance: f32,
    selection: &baux::semantics::optimizer::surrogate::PoolSelection,
) -> Result<StepDecision, String> {
    let improvement = f64::from(value) - f64::from(selection.baseline_value);
    let predicted = f64::from(selection.candidate.mean) - f64::from(selection.baseline_value);
    let obs_var = f64::from(variance);
    let threshold = 0.0;
    super::controller::validate_improvements(improvement, predicted)?;
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
    Ok(StepDecision {
        numerical,
        accepted,
        improvement,
        threshold,
        predicted_improvement: predicted,
        action: RadiusAction::Hold,
        next_radius: ctrl.radius(),
        agreement_ratio: (predicted > 0.0).then_some(improvement / predicted),
    })
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
}
