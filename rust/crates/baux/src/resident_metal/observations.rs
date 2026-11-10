//! Acceptance policies for already-evaluated proposals.
//!
//! All control values are rewards (higher is better), and variance arguments
//! are variances of the reported estimates, not standard deviations. Evaluation
//! is the caller's responsibility; none of these methods runs a forward pass.
//!
//! The policies have different baselines: initialization and `tell_noisy` use
//! stored incumbent values, `tell_modeled` uses an ENN incumbent prediction,
//! and `paired_*` uses a caller-measured same-condition difference. They are
//! not interchangeable estimators. Calibration multipliers are empirical,
//! not certified confidence levels or guarantees of training progress.
//!
//! Accepted and rejected trials both enter bounded history through `history`.
//! Successful tell applies any accepted weights, clears the pending proposal,
//! and queues the decision for `sync`; it is not merely a telemetry operation.

use super::*;
use baux::semantics::numerical::{self, Kind, Trace};

impl SearchState {
    /// Scalar-reward adapter for `initial_objectives` during ENN warm-up.
    /// The pending trial must come from the same search state; variance is the
    /// estimate variance. This records the observation and may accept weights.
    pub fn tell_initial(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.initial_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Record a scalar/vector warm-up observation before the configured count
    /// is reached. Vector control may override scalar acceptance;
    /// initialization is not an unconditional commit and does not use
    /// normal acquisition gain.
    pub fn initial_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        self.check_unleased()?;
        self.initial_observation(round, observation)
    }

    /// Implement warm-up acceptance against the stored incumbent.
    /// Scalar initialization requires positive reward gain (`threshold = 0.0`,
    /// matching upstream `enn`). Records history without normal radius
    /// adaptation, and initializes controller state when warm-up completes.
    fn initial_observation(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            if !self.implicit_history
                || self.initial_observations == 0
                || self.history >= self.initial_observations
            {
                return Err("Metal ENN initialization is complete or not configured".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let control_threshold = 0.0;
            let objectives = self.objective_decision(observation)?;
            let (accepted, improvement) = objectives.unwrap_or((
                baux::semantics::optimizer::accept_candidate(
                    control_improvement,
                    control_threshold,
                ),
                control_improvement,
            ));
            let numerical = if objectives.is_some() {
                numerical::unchecked("vector objective override")?
            } else {
                numerical::check(Trace::observed(
                    Kind::Paired,
                    control_improvement,
                    0.0,
                    f64::from(variance),
                    0.0,
                    control_threshold,
                    accepted,
                ))?
            };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                None,
                None,
                false,
            )?;
            if self.history == self.initial_observations && !self.has_morbo() {
                self.trust.restart();
                self.trust_observations = self.history;
                self.trust.seed_baseline(f64::from(self.best), self.history);
                self.length = self.trust.length();
            }
            Ok(NoisyDecision {
                numerical,
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold: control_threshold,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome: if accepted {
                    TrustRegionOutcome::Success
                } else {
                    TrustRegionOutcome::Failure
                },
            })
        })
    }

    /// Scalar adapter for stored-incumbent acceptance (`threshold = 0.0`,
    /// matching upstream `enn`).
    pub fn tell_noisy(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.noisy_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Compare a new observation with stored incumbent reward metadata.
    /// Scalar acceptance requires positive reward gain (`threshold = 0.0`,
    /// matching upstream `enn`); vector control may override it. Records the
    /// trial and adapts the region.
    pub fn noisy_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            self.check_unleased()?;
            self.check_round(round)?;
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let control_threshold = 0.0;
            let objectives = self.objective_decision(observation)?;
            let (accepted, improvement, threshold) = objectives.map_or(
                (
                    baux::semantics::optimizer::accept_candidate(
                        control_improvement,
                        control_threshold,
                    ),
                    control_improvement,
                    control_threshold,
                ),
                |(accepted, improvement)| (accepted, improvement, 0.0),
            );
            let numerical = if objectives.is_some() {
                numerical::unchecked("vector objective override")?
            } else {
                numerical::check(Trace::observed(
                    Kind::Combined,
                    control_improvement,
                    0.0,
                    f64::from(variance),
                    f64::from(incumbent_variance),
                    control_threshold,
                    accepted,
                ))?
            };
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else {
                TrustRegionOutcome::Failure
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                Some(trust_outcome),
                true,
            )?;
            Ok(NoisyDecision {
                numerical,
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome,
            })
        })
    }

    /// Scalar adapter for ENN-predicted-incumbent acceptance.
    ///
    /// Evaluates whether a proposed parameter candidate improves upon the
    /// current incumbent using an Epistemic Nearest Neighbors (ENN) posterior
    /// prediction of the incumbent rather than an expensive contemporaneous
    /// forward pass on the incumbent.
    ///
    /// # Units
    ///
    /// * `value`: Negative negative-log-likelihood (-NLL) in nats/token (or
    ///   scaled nats/token reward; higher is better).
    /// * `variance`: Variance of the estimated reward in (nats/token)²,
    ///   representing aleatoric or batch estimation uncertainty. Must be finite
    ///   and non-negative.
    ///
    /// # Baseline Comparison & Threshold
    ///
    /// Unlike [`tell_noisy`](Self::tell_noisy), which compares against a
    /// recorded historical observation of the incumbent, `tell_modeled`
    /// compares against `round.incumbent_mean`, which is an ENN posterior
    /// prediction generated during ask time.
    ///
    /// * **Control Improvement**: $$\Delta = \text{value} -
    ///   \text{round.incumbent\_mean}$$ measured in nats/token.
    /// * **Incumbent Uncertainty**: $$\sigma_{\text{inc}} =
    ///   \text{round.incumbent\_standard\_error}$$ measured in nats/token.
    /// * **Control Threshold**: `control_threshold = 0.0` (standard zero-margin
    ///   TuRBO acceptance).
    ///
    /// A candidate is accepted when $\Delta > \tau$.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if:
    /// * The search state is in paired-relative observation mode
    ///   (`self.relative == true`).
    /// * Any of `value`, `variance`, `round.predicted_mean`,
    ///   `round.incumbent_mean`, or their associated standard errors are
    ///   non-finite (NaN or infinite) or negative variances.
    /// * `round` token does not match the pending proposal state.
    ///
    /// See [`modeled_objectives`](Self::modeled_objectives) for detailed trust
    /// region updates and side effects.
    pub fn tell_modeled(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.modeled_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Bind the pending materialized candidate to its evaluation condition.
    /// Reissuing a plan is allowed only for the same world (e.g. after a scorer
    /// error); a completed or foreign proposal cannot obtain one.
    pub fn plan_evaluation(
        &mut self,
        round: &Proposals,
        world: baux::semantics::optimizer::WorldId,
    ) -> Result<baux::semantics::optimizer::AskPlan, String> {
        self.check_unleased()?;
        self.check_round(round)?;
        if self.base_id != round.base_id || self.evaluation.is_some_and(|planned| planned != world)
        {
            return Err("evaluation plan changed incumbent or world".into());
        }
        self.evaluation = Some(world);
        Ok(baux::semantics::optimizer::AskPlan::new(
            baux::semantics::optimizer::RoundId {
                owner: round.owner,
                proposal: round.id,
                incumbent: round.base_id,
            },
            world,
        ))
    }

    /// Consume one candidate-bound scalar evaluation. Warm-up and modeled
    /// acceptance keep their existing arithmetic; neither certifies
    /// calibration.
    pub fn tell_evaluated(
        &mut self,
        round: &Proposals,
        evaluation: baux::semantics::optimizer::Evaluated,
    ) -> Result<NoisyDecision, String> {
        let receipt = evaluation.receipt();
        let value = -receipt.candidate_loss as f32;
        let variance = receipt.candidate_variance as f32;
        check_scores(&[value], &[variance])?;
        self.evaluated_objectives(
            round,
            evaluation,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Consume one candidate-bound evaluation with a direct-sum vector
    /// observation.
    pub fn evaluated_objectives(
        &mut self,
        round: &Proposals,
        evaluation: baux::semantics::optimizer::Evaluated,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        use baux::semantics::optimizer::{RoundId, TellDecision};
        self.check_unleased()?;
        self.check_round(round)?;
        // Public proposal metadata is inspectable/mutable by callers. Only the
        // saved publication describes the materialized row and its prediction.
        let published = self.pending.as_ref().ok_or("No pending proposal")?.clone();
        let active = RoundId {
            owner: self.owner,
            proposal: round.id,
            incumbent: self.base_id,
        };
        let candidate = self
            .observation
            .checked_add(1)
            .ok_or("Observation ID overflow")?;
        let receipt = evaluation.receipt();
        if TellDecision::new(active, evaluation.round(), candidate, false).is_none()
            || self.evaluation != Some(receipt.world)
            || receipt.paired_gain.is_some()
        {
            return Err("evaluation does not match the pending scalar round".into());
        }
        let value = -receipt.candidate_loss as f32;
        let variance = receipt.candidate_variance as f32;
        check_scores(&[value], &[variance])?;
        if observation.control().mean.to_bits() != value.to_bits()
            || observation.control().variance.to_bits() != variance.to_bits()
        {
            return Err("evaluation receipt does not match observation control".into());
        }
        if self.implicit_history && self.history < self.initial_observations {
            self.initial_objectives(&published, observation)
        } else {
            self.modeled_objectives(&published, observation)
        }
    }

    /// Accept against the ENN incumbent prediction carried by the pending ask.
    ///
    /// Decides acceptance of an evaluated candidate against the Epistemic
    /// Nearest Neighbors (ENN) posterior prediction of the incumbent,
    /// supporting either scalar or multi-objective vector observations.
    ///
    /// # Units and Semantics
    ///
    /// * `observation.control().mean` (`value`): Scalar reward in negative NLL
    ///   (nats/token, higher is better).
    /// * `observation.control().variance` (`variance`): Evaluation variance in
    ///   (nats/token)².
    /// * `round.incumbent_mean`: Predicted incumbent mean reward in nats/token,
    ///   evaluated via posterior acquisition in Metal (`acquisition_score` with
    ///   `incumbent_index` excluded as a leave-one-out query).
    /// * `round.incumbent_standard_error`: Incumbent standard error in
    ///   nats/token, combining epistemic and aleatoric posterior uncertainty.
    ///
    /// # Improvement and Threshold Calculation
    ///
    /// For scalar control:
    /// 1. **Control Improvement**: $$\text{control\_improvement} = \text{value}
    ///    - \text{round.incumbent\_mean}$$ represents the scalar gain over the
    ///    predicted incumbent baseline in nats/token.
    /// 2. **Control Threshold**: `control_threshold = 0.0` (standard
    ///    zero-margin TuRBO acceptance).
    /// 3. **Acceptance Rule**: The proposal is accepted if
    ///    $\text{control\_improvement} > \text{control\_threshold}$ (or as
    ///    dictated by vector Pareto / multi-objective criteria if
    ///    `observation.is_vector()`).
    ///
    /// # State and Side Effects
    ///
    /// When called:
    /// * Validates finite values and positive variances across candidate and
    ///   incumbent parameters.
    /// * Updates trust region outcome (`Success` if accepted, `Failure` if
    ///   improvement < -threshold, else `Inconclusive`).
    /// * Commits accepted candidate weights or retains the incumbent weights.
    /// * Records evidence to bounded history.
    /// * Updates posterior calibration multipliers for subsequent optimization
    ///   rounds.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if:
    /// * The search state is configured for paired-relative observations.
    /// * Values or variances are NaN, infinite, or variances are negative.
    /// * The proposal round does not match the active session.
    pub fn modeled_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<NoisyDecision, String> {
        let crate::objective_observation::ObjectiveEstimate {
            mean: value,
            variance,
        } = observation.control();
        autoreleasepool(|| {
            self.check_unleased()?;
            self.check_round(round)?;
            check_scores(
                &[value, round.predicted_mean, round.incumbent_mean],
                &[
                    variance,
                    round.predicted_standard_error.powi(2),
                    round.incumbent_standard_error.powi(2),
                ],
            )?;
            let incumbent_value = round.incumbent_mean;
            let incumbent_variance = round.incumbent_standard_error.powi(2);
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let control_threshold = 0.0;
            let objectives = self.objective_decision(observation)?;
            let (accepted, improvement, threshold) = objectives.map_or(
                (
                    baux::semantics::optimizer::accept_candidate(
                        control_improvement,
                        control_threshold,
                    ),
                    control_improvement,
                    control_threshold,
                ),
                |(accepted, improvement)| (accepted, improvement, 0.0),
            );
            let numerical = if objectives.is_some() {
                numerical::unchecked("vector objective override")?
            } else {
                numerical::check(Trace::observed(
                    Kind::Separate,
                    control_improvement,
                    0.0,
                    f64::from(variance),
                    f64::from(incumbent_variance),
                    control_threshold,
                    accepted,
                ))?
            };
            let control_prediction = f64::from(round.predicted_mean) - f64::from(incumbent_value);
            let predicted_improvement = (!self.has_morbo()).then_some(control_prediction);
            let agreement_ratio = predicted_improvement
                .filter(|prediction| *prediction > f64::EPSILON)
                .map(|prediction| improvement / prediction);
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else {
                TrustRegionOutcome::Failure
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                Some(trust_outcome),
                true,
            )?;
            Ok(NoisyDecision {
                numerical,
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement,
                agreement_ratio,
                trust_outcome,
            })
        })
    }
}
