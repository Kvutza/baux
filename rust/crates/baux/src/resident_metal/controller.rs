//! Refit ENN and advance radius controllers after recording observations.
//!
//! Fitting changes estimator scales and, when enabled, tensor-family geometry;
//! controller updates change the dimensionless radius used by subsequent asks.
//! Neither operation evaluates the objective or accepts a new parameter row.
//! The caller has already made that decision in `observations`/`history`.
//! Controller restarts retain the current incumbent, not a new random model.

use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

impl SearchState {
    /// Consume supplied reliability evidence and copy back the next radius.
    /// Increments the trust observation counter and errors if no reliability
    /// controller exists. Evidence is computed elsewhere; this is not a new
    /// measurement or a proof that a radius change improves learning.
    pub(super) fn update_reliability(
        &mut self,
        evidence: reliability_region::ReliabilityEvidence,
    ) -> Result<(), String> {
        self.trust_observations += 1;
        let controller = self
            .reliability
            .as_mut()
            .ok_or("Reliability controller is not configured")?;
        controller.update(evidence)?;
        self.length = controller.length();
        Ok(())
    }

    /// Refit estimator scales from the bounded distance/outcome/variance
    /// tables. Requires initialized fitter parameters. Optional
    /// family-shape selection runs at warm-up completion and every fourth
    /// later observation; it uses CPU scoring below 64 retained rows and
    /// GPU scoring otherwise. ENN fitting itself runs on every call with a
    /// reproducible seed based on history size. Stores fitted k/scales for
    /// the next ask; query-local risk-prefix selection is a separate kernel
    /// policy. Does not certify calibration or alter weights.
    pub(super) fn fit_enn(&mut self) -> Result<(), String> {
        let n = self.history;
        let cadence = (self.observation as usize).max(n);
        let fit_family = cadence == self.initial_observations
            || (cadence > self.initial_observations
                && (cadence - self.initial_observations) % 4 == 0);
        if fit_family {
            if let Some(family) = &mut self.family {
                let outcomes =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.outcomes[i]));
                let variances =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.variances[i]));
                let params = *self
                    .fitter
                    .as_ref()
                    .and_then(|f| f.params())
                    .ok_or("Family fit needs ENN parameters")?;
                let candidates = family.candidates();
                // Paired warm measurements cross over at H=64 on this Apple
                // GPU. A tiny history must not pay more launch
                // latency than CPU work.
                let local = (self.distance_scaling != crate::config::DistanceScaling::Global)
                    .then_some(self.local_scale_neighbors);
                let scores = if n < 64 {
                    family.reference(
                        &candidates,
                        &outcomes.view(),
                        &variances.view(),
                        params,
                        self.fit_samples,
                        self.fit_seed.wrapping_add(n as u64),
                        local,
                    )?
                } else {
                    self.metric_gpu
                        .as_mut()
                        .ok_or("Family metric GPU is missing")?
                        .score(
                            family,
                            &candidates,
                            &outcomes.view(),
                            &variances.view(),
                            params,
                            self.fit_samples,
                            self.fit_seed.wrapping_add(n as u64),
                            local,
                        )?
                };
                family.select(&candidates, &scores, self.fit_samples.min(n))?;
                self.apply_family()?;
            }
        }
        let raw_distances = ndarray::Array2::from_shape_fn((n, n), |(row, column)| {
            f64::from(self.total_distance(row, column))
        });
        let distances = match self.distance_scaling {
            crate::config::DistanceScaling::SelfTuning => {
                crate::fit::tuned_distances(&raw_distances.view(), self.local_scale_neighbors)
                    .map_err(|error| error.to_string())?
            }
            crate::config::DistanceScaling::Hyperbolic => {
                crate::fit::hyperbolic_distances(&raw_distances.view(), self.local_scale_neighbors)
                    .map_err(|error| error.to_string())?
            }
            crate::config::DistanceScaling::Global => raw_distances,
        };
        let outcomes =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.outcomes[row]));
        let variances =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.variances[row]));
        let mut rng = StdRng::seed_from_u64(self.fit_seed.wrapping_add(n as u64));
        let fitter = self
            .fitter
            .as_mut()
            .ok_or("Metal implicit ENN is not configured")?;
        // Fit the variance scales at the configured neighbor count. Query-local
        // prefix selection is an independent selection-kernel policy.
        let params = fitter
            .ask_distances(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
            .map_err(|error| error.to_string())?;
        let y_scale = fitter.y_std()[0];
        self.fitted_enn = Some((
            usize::try_from(params.k_neighbors)
                .map_err(|_| "Fitted ENN neighbor count does not fit usize")?,
            params.epistemic_scale as f32,
            params.aleatoric_scale as f32,
            y_scale as f32,
        ));
        Ok(())
    }

    /// Update ordinary TuRBO from a reward used as its own batch baseline.
    /// Clears accumulated paired evidence. On restart, retains the incumbent
    /// row, resets trust counters, and collapses only non-implicit logical
    /// history. Copies controller length back without evaluating or
    /// randomizing a model.
    pub(super) fn update_region(&mut self, value: f32) -> Result<(), String> {
        self.paired_evidence_sum = 0.0;
        self.paired_evidence_count = 0;
        let value = f64::from(value);
        self.trust_observations += 1;
        self.trust
            .update_batch(
                &ndarray::ArrayView1::from(&[value]),
                self.trust_observations,
                value,
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.set_watermark(0);
            let best = f64::from(self.best);
            self.trust
                .update_batch(&ndarray::ArrayView1::from(&[best]), 1, best)
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    /// Update ordinary TuRBO against a caller-supplied incumbent reward.
    /// Clears paired evidence and follows the same incumbent-preserving restart
    /// bookkeeping as `update_region`; no estimate variance is consumed here.
    pub(super) fn update_noisy(&mut self, value: f32, incumbent: f32) -> Result<(), String> {
        self.paired_evidence_sum = 0.0;
        self.paired_evidence_count = 0;
        let value = f64::from(value);
        self.trust_observations += 1;
        self.trust
            .update_batch(
                &ndarray::ArrayView1::from(&[value]),
                self.trust_observations,
                f64::from(incumbent),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.set_watermark(0);
            let best = f64::from(self.best);
            self.trust
                .update_batch(&ndarray::ArrayView1::from(&[best]), 1, best)
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    /// Update ordinary TuRBO using an explicitly classified trust outcome.
    /// The caller supplies Success/Failure/Inconclusive; this method does not
    /// reclassify uncertainty. On restart retains incumbent weights, preserves
    /// implicit history, and reseeds the controller with an inconclusive
    /// baseline.
    pub(super) fn update_outcome(
        &mut self,
        value: f32,
        incumbent: f32,
        outcome: TrustRegionOutcome,
    ) -> Result<(), String> {
        self.trust_observations += 1;
        let y_new = [f64::from(value)];
        self.trust
            .update_outcome(
                &ndarray::ArrayView1::from(&y_new),
                self.trust_observations,
                f64::from(incumbent),
                outcome,
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.set_watermark(0);
            let seed = [f64::from(self.best)];
            self.trust
                .update_outcome(
                    &ndarray::ArrayView1::from(&seed),
                    1,
                    f64::from(self.best),
                    TrustRegionOutcome::Inconclusive,
                )
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }
}
