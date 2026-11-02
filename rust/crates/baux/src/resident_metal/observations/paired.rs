use super::*;
use baux::semantics::numerical::{self, Kind, Trace};

impl SearchState {
    /// Accumulate inconclusive signed standardized differences for radius
    /// control. A negative boundary crossing becomes Failure; a positive
    /// crossing stays Inconclusive and cannot accept weights or expand the
    /// region by itself. Decisive local outcomes reset the accumulator.
    /// This heuristic does not establish independence of successive
    /// differences or a sequential test bound.
    fn pair_outcome(
        &mut self,
        local: TrustRegionOutcome,
        improvement: f64,
        variance: f64,
        boundary: f64,
    ) -> TrustRegionOutcome {
        if local != TrustRegionOutcome::Inconclusive {
            self.paired_evidence_sum = 0.0;
            self.paired_evidence_count = 0;
            return local;
        }
        let standard_error = variance.sqrt();
        if !standard_error.is_finite() || standard_error <= f64::EPSILON {
            return local;
        }
        self.paired_evidence_sum += improvement / standard_error;
        self.paired_evidence_count += 1;
        let accumulated = self.paired_evidence_sum / (self.paired_evidence_count as f64).sqrt();
        if accumulated.abs() < boundary {
            return local;
        }
        self.paired_evidence_sum = 0.0;
        self.paired_evidence_count = 0;
        if accumulated < 0.0 {
            TrustRegionOutcome::Failure
        } else {
            // Positive aggregate evidence does not expand the region without an
            // accepted incumbent update. Clear it symmetrically and wait for a
            // decisive candidate.
            TrustRegionOutcome::Inconclusive
        }
    }

    /// Apply a caller-decided acceptance using supplied absolute measurements.
    /// Does not compute a paired difference or threshold itself. Records the
    /// trial and delegates region adaptation without model-aware reliability
    /// evidence; use a modeled paired API when that evidence is required.
    pub fn tell_paired(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
    ) -> Result<(), String> {
        baux::semantics::numerical::unchecked("caller-decided paired acceptance")?;
        autoreleasepool(|| {
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accept,
                None,
                None,
                true,
            )
        })
    }

    /// Record an outcome and decide from a supplied same-condition difference.
    /// `value`/`variance` describe the outcome to store; `paired_improvement`
    /// and its variance control acceptance. They may be on different anchored
    /// reporting scales. This method does not measure either model or fit a
    /// response posterior; it delegates to `paired_observation` without blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn paired_modeled(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        paired_improvement: f32,
        improvement_variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.paired_observation(
            round,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            paired_improvement,
            improvement_variance,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Like `paired_modeled`, with per-block differences for reliability
    /// evidence. The blocks must come from the caller's paired evaluation;
    /// they are not additional function evaluations or observations at
    /// other parameter states.
    pub fn paired_blocks(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        paired_improvement: f32,
        improvement_variance: f32,
        blocks: &[f32],
    ) -> Result<NoisyDecision, String> {
        self.paired_observation(
            round,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            paired_improvement,
            improvement_variance,
            Some(blocks),
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Implement acceptance from the caller's measured paired reward
    /// difference. Requires finite values and nonnegative estimate
    /// variances. Accepts only above `past_multiplier *
    /// sqrt(improvement_variance)`, builds calibration and reliability
    /// evidence, and records/commits through `tell_absolute`. Inconclusive
    /// differences can affect radius evidence but cannot accept
    /// weights through accumulation alone. No concentration theorem is assumed.
    fn paired_observation(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        paired_improvement: f32,
        improvement_variance: f32,
        _blocks: Option<&[f32]>,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            self.check_round(round)?;
            let pred_var = round.predicted_standard_error.powi(2);
            let inc_var = round.incumbent_standard_error.powi(2);
            check_scores(
                &[
                    value,
                    incumbent_value,
                    round.predicted_mean,
                    round.incumbent_mean,
                ],
                &[variance, incumbent_variance, pred_var, inc_var],
            )?;
            if !paired_improvement.is_finite()
                || !improvement_variance.is_finite()
                || improvement_variance < 0.0
            {
                return Err(
                    "Paired improvement must be finite and variance must be finite and nonnegative"
                        .into(),
                );
            }
            let improvement = f64::from(paired_improvement);
            let multiplier = f64::from(self.calibration.multiplier());
            let threshold = multiplier * f64::from(improvement_variance).sqrt();
            let accepted = baux::semantics::optimizer::accept_candidate(improvement, threshold);
            let numerical = numerical::check(Trace::observed(
                Kind::Paired,
                improvement,
                multiplier,
                f64::from(improvement_variance),
                0.0,
                threshold,
                accepted,
            ))?;
            let predicted_improvement =
                f64::from(round.predicted_mean) - f64::from(round.incumbent_mean);
            let calibration_scale = (f64::from(improvement_variance)
                + f64::from(round.predicted_standard_error).powi(2)
                + f64::from(round.incumbent_standard_error).powi(2))
            .sqrt();
            let mut calibration = self.calibration.clone();
            if self.history >= self.initial_observations {
                calibration.observe(
                    paired_improvement,
                    predicted_improvement as f32,
                    calibration_scale as f32,
                )?;
            }
            let agreement_ratio = (predicted_improvement > f64::EPSILON)
                .then_some(improvement / predicted_improvement);
            let local_trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let trust_outcome = self.pair_outcome(
                local_trust_outcome,
                improvement,
                f64::from(improvement_variance),
                multiplier,
            );
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(incumbent_value),
                Some(trust_outcome),
                true,
            )?;
            self.calibration = calibration;
            Ok(NoisyDecision {
                numerical,
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: Some(predicted_improvement),
                agreement_ratio,
                trust_outcome,
            })
        })
    }
}
