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
                None,
                true,
            )
        })
    }

    /// Scalar-reward adapter for `initial_objectives` during ENN warm-up.
    /// The pending trial must come from the same search state; variance is the
    /// estimate variance. This records the observation and may accept weights.
    pub(crate) fn tell_initial(
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
        self.initial_observation(round, observation)
    }

    /// Implement warm-up acceptance against the stored incumbent.
    /// Fresh scalar initialization requires positive reward gain; rewarming
    /// uses two combined standard errors. Records history without normal
    /// radius adaptation, and initializes controller state when warm-up
    /// completes.
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
            let rewarming = self.observation as usize > self.history;
            let pair_variance = if variance.to_bits() == incumbent_variance.to_bits() {
                0.0
            } else {
                f64::from(variance)
            };
            let control_threshold = if rewarming {
                2.0 * (f64::from(variance) + f64::from(incumbent_variance)).sqrt()
            } else {
                f64::from(self.calibration.multiplier()) * pair_variance.sqrt()
            };
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
                    if rewarming {
                        Kind::Combined
                    } else {
                        Kind::Paired
                    },
                    control_improvement,
                    if rewarming {
                        2.0
                    } else {
                        f64::from(self.calibration.multiplier())
                    },
                    if rewarming {
                        f64::from(variance)
                    } else {
                        pair_variance
                    },
                    if rewarming {
                        f64::from(incumbent_variance)
                    } else {
                        0.0
                    },
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
                None,
                false,
            )?;
            if self.history == self.initial_observations && !self.has_morbo() {
                if let Some(controller) = &self.reliability {
                    self.length = controller.length();
                } else {
                    let failure_tolerance = self.trust.failure_tolerance() as usize;
                    self.trust = TurboTrustRegion::new(self.dimensions, self.length_config);
                    self.trust.set_arms(1);
                    self.trust
                        .set_tolerance(failure_tolerance)
                        .map_err(|error| error.to_string())?;
                    let observed = self.outcomes[..self.history]
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>();
                    self.trust_observations = observed.len();
                    self.trust
                        .update(&ndarray::ArrayView1::from(&observed), observed.len())
                        .map_err(|error| error.to_string())?;
                    self.length = self.trust.length();
                }
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
                trust_outcome: TrustRegionOutcome::Inconclusive,
            })
        })
    }

    /// Scalar adapter for stored-incumbent noisy acceptance.
    /// Uses `2 * sqrt(candidate_variance + incumbent_variance)` as its
    /// threshold. No same-condition incumbent measurement or covariance is
    /// computed here; the caller is responsible for interpreting the
    /// supplied estimate errors.
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

    /// Compare a new observation with stored incumbent reward/error metadata.
    /// Scalar acceptance requires gain above two combined standard errors;
    /// vector control may override it. Records the trial and adapts the region.
    /// Unlike `modeled_objectives`, it supplies no reliability-model evidence.
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
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let control_improvement = f64::from(value) - f64::from(incumbent_value);
            let combined_variance = f64::from(variance) + f64::from(incumbent_variance);
            if !combined_variance.is_finite() {
                return Err("Combined noisy-observation variance is not finite".into());
            }
            let control_threshold = 2.0 * combined_variance.sqrt();
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
                    2.0,
                    f64::from(variance),
                    f64::from(incumbent_variance),
                    control_threshold,
                    accepted,
                ))?
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                None,
                None,
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
                trust_outcome: if accepted {
                    TrustRegionOutcome::Success
                } else {
                    TrustRegionOutcome::Failure
                },
            })
        })
    }

    /// Scalar adapter for ENN-predicted-incumbent acceptance.
    ///
    /// Evaluates whether a proposed parameter candidate improves upon the
    /// current incumbent using an Epistemic Neural Network (ENN) posterior
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
    /// * **Control Threshold**: $$\tau = c \cdot \left(\sqrt{\text{variance}} +
    ///   \sigma_{\text{inc}}\right)$$ where $c$ is the calibration multiplier
    ///   (`self.calibration.multiplier()`).
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
    /// Neural Network (ENN) posterior prediction of the incumbent,
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
    /// 2. **Control Threshold**: $$\text{control\_threshold} = c \cdot
    ///    \left(\sqrt{\text{variance}} +
    ///    \text{round.incumbent\_standard\_error}\right)$$ where $c$ is the
    ///    historical calibration multiplier (`self.calibration.multiplier()`).
    ///    The threshold scales with the sum of observation standard error and
    ///    incumbent posterior standard error.
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
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
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
            let control_threshold = if self.reliability.is_some() {
                f64::from(self.calibration.multiplier())
                    * (f64::from(variance).sqrt() + f64::from(incumbent_variance).sqrt())
            } else {
                0.0
            };
            let calibration =
                self.next_calibration(round, value, variance, self.objective_selector.is_some())?;
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
                    f64::from(self.calibration.multiplier()),
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
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            let reliability = if self.has_morbo() {
                None
            } else {
                self.reliability_evidence(
                    round,
                    improvement,
                    f64::from(variance) + f64::from(incumbent_variance),
                    accepted,
                    None,
                )?
            };
            self.absolute_objectives(
                round,
                observation,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                Some(trust_outcome),
                reliability,
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
                predicted_improvement,
                agreement_ratio,
                trust_outcome,
            })
        })
    }

    /// Prepare, without installing, calibration after the current observation.
    /// Only scalar observations after warm-up contribute a standardized
    /// residual against the ask-time candidate prediction. Current
    /// acceptance still uses the previous calibration state, avoiding use
    /// of its own residual threshold.
    fn next_calibration(
        &self,
        round: &Proposals,
        value: f32,
        variance: f32,
        vector: bool,
    ) -> Result<crate::calibration::Calibration, String> {
        let mut next = self.calibration.clone();
        if self.history >= self.initial_observations && !vector {
            next.observe(
                value,
                round.predicted_mean,
                (variance + round.predicted_standard_error.powi(2)).sqrt(),
            )?;
        }
        Ok(next)
    }

    /// Record an outcome and decide from a supplied same-condition difference.
    /// `value`/`variance` describe the outcome to store; `paired_improvement`
    /// and its variance control acceptance. They may be on different anchored
    /// reporting scales. This method does not measure either model or fit a
    /// response posterior; it delegates to `paired_observation` without blocks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paired_modeled(
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
    pub(crate) fn paired_blocks(
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
        blocks: Option<&[f32]>,
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
            let reliability = self.reliability_evidence(
                round,
                improvement,
                f64::from(improvement_variance),
                accepted,
                blocks,
            )?;
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(incumbent_value),
                Some(trust_outcome),
                reliability,
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
