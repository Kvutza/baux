//! Refit ENN and advance radius controllers after recording observations.
//!
//! Fitting changes estimator scales and, when enabled, tensor-family geometry;
//! controller updates change the dimensionless radius used by subsequent asks.
//! Neither operation evaluates the objective or accepts a new parameter row.
//! The caller has already made that decision in `observations`/`history`.
//! Controller restarts retain the current incumbent, not a new random model.

use super::*;
use crate::hash::prng;

impl SearchState {
    pub fn fit_enn(&mut self) -> Result<(), String> {
        let n = self.history;
        let raw_distances = ndarray::Array2::from_shape_fn((n, n), |(row, column)| {
            f64::from(self.total_distance(row, column))
        });
        let distances = match self.distance_metric {
            crate::config::DistanceMetric::Conformal => {
                crate::fit::conformal_distances(&raw_distances.view(), self.local_scale_neighbors)
                    .map_err(|error| error.to_string())?
            }
            crate::config::DistanceMetric::Hyperbolic => {
                crate::fit::hyperbolic_distances(&raw_distances.view(), self.local_scale_neighbors)
                    .map_err(|error| error.to_string())?
            }
            crate::config::DistanceMetric::Wasserstein => {
                crate::fit::wasserstein_distances(&raw_distances.view(), self.local_scale_neighbors)
                    .map_err(|error| error.to_string())?
            }
            crate::config::DistanceMetric::Euclidean => raw_distances,
        };
        let outcomes =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.outcomes[row]));
        let variances =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.variances[row]));
        let mut rng = prng(self.fit_seed.wrapping_add(n as u64));
        let fitter = self
            .fitter
            .as_mut()
            .ok_or("Metal implicit ENN is not configured")?;
        let params = if self.adapt_neighbors {
            fitter.ask_adaptive(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                self.initial_observations as i32,
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        } else {
            fitter.ask_distances(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        }
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
    pub fn update_region(&mut self, value: f32) -> Result<(), String> {
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
            if !self.history_rows.is_empty() {
                self.copy(&self.base, &self.history_rows[0])?;
                self.resident_history = 1;
            }
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.pairwise_distances.fill(0.0);
                self.fitted_enn = None;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.seed_baseline(f64::from(self.best), 1);
        }
        self.length = self.trust.length();
        Ok(())
    }

    /// Update ordinary TuRBO against a caller-supplied incumbent reward.
    /// Follows the same incumbent-preserving restart
    /// bookkeeping as `update_region`; no estimate variance is consumed here.
    pub fn update_noisy(&mut self, value: f32, incumbent: f32) -> Result<(), String> {
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
            if !self.history_rows.is_empty() {
                self.copy(&self.base, &self.history_rows[0])?;
                self.resident_history = 1;
            }
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.pairwise_distances.fill(0.0);
                self.fitted_enn = None;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.seed_baseline(f64::from(self.best), 1);
        }
        self.length = self.trust.length();
        Ok(())
    }

    /// Update ordinary TuRBO using an explicitly classified trust outcome.
    /// The caller supplies Success/Failure/Inconclusive; this method does not
    /// reclassify uncertainty. On restart retains incumbent weights, preserves
    /// implicit history, and reseeds the controller with an inconclusive
    /// baseline.
    pub fn update_outcome(
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
            if !self.history_rows.is_empty() {
                self.copy(&self.base, &self.history_rows[0])?;
                self.resident_history = 1;
            }
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
                self.pairwise_distances.fill(0.0);
                self.fitted_enn = None;
                self.objective_history.retain_incumbent();
                self.retain_axis();
            }
            self.trust_observations = 1;
            self.trust.seed_baseline(f64::from(self.best), 1);
        }
        self.length = self.trust.length();
        Ok(())
    }
}
