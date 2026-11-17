//! Refit ENN and advance radius controllers after recording observations.

use super::*;
use crate::hash::prng;

impl SearchState {
    fn cow_slot(&mut self, slot: usize) -> Result<(), String> {
        if std::ptr::eq::<metal::BufferRef>(&**self.history_rows[slot], &**self.base)
            || self.history_rows.iter().enumerate().any(|(i, row)| {
                i != slot && std::ptr::eq::<metal::BufferRef>(&***row, &**self.history_rows[slot])
            })
        {
            self.history_rows[slot] = Arc::new(self.runtime.buffer::<u16>(self.dimensions));
        }
        Self::cow_buffer(&self.runtime, &mut self.history_rows[slot])
    }

    pub fn fit_enn(mut self) -> Result<Self, String> {
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
        let scalar_y_bounds = self
            .manifold
            .as_ref()
            .and_then(|m| m.config.y_bounds.as_ref())
            .and_then(|b| b.first().copied());
        let mut outcomes = ndarray::Array2::<f64>::zeros((n, 1));
        let mut variances = ndarray::Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            let raw_v = if self.noise_aware {
                self.variances[row]
            } else {
                0.0
            };
            let (wy, wv) = manifold::warp_observation(self.outcomes[row], raw_v, scalar_y_bounds);
            outcomes[[row, 0]] = f64::from(wy);
            variances[[row, 0]] = f64::from(wv);
        }
        let yvar_view = self.noise_aware.then(|| variances.view());
        let mut rng = prng(self.fit_seed.wrapping_add(n as u64));
        let fitter = self
            .fitter
            .take()
            .ok_or("Metal implicit ENN is not configured")?;
        let (fitter, params) = if self.adapt_neighbors {
            fitter.ask_adaptive(
                &distances.view(),
                &outcomes.view(),
                yvar_view.as_ref(),
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
                yvar_view.as_ref(),
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
        self.fitter = Some(fitter);
        Ok(self)
    }

    fn restart_bookkeeping(&mut self) -> Result<(), String> {
        self.trust.restart();
        self.restart_count += 1;
        if !self.history_rows.is_empty() {
            self.cow_slot(0)?;
            self.copy(&*self.base, &*self.history_rows[0])?;
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
        Ok(())
    }

    fn adapt_tolerance(&mut self, improved: Option<bool>, success_tol: i32) -> bool {
        let mut len = self.length;
        let mut sc = self.trust.success_counter();
        let mut fc = self.trust.failure_counter();
        let ft = self.trust.failure_tolerance();
        let _ = baux::semantics::optimizer::turbo_adapt(
            &mut len,
            self.length_config.length_max,
            &mut sc,
            &mut fc,
            success_tol,
            ft,
            improved,
        );
        self.length = len;
        len < self.length_config.length_min
    }

    pub fn update_region(mut self, value: f32) -> Result<Self, String> {
        let value = f64::from(value);
        self.trust_observations += 1;
        if let Some(st) = self.success_tolerance_override
            && st != 3
        {
            let needs_restart = self.adapt_tolerance(Some(false), st);
            if needs_restart {
                self.restart_bookkeeping()?;
                self.length = self.length_config.length_init;
            }
            return Ok(self);
        }
        self.trust
            .update_batch(
                &ndarray::ArrayView1::from(&[value]),
                self.trust_observations,
                value,
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.restart_bookkeeping()?;
        }
        self.length = self.trust.length();
        Ok(self)
    }

    pub fn update_noisy(mut self, value: f32, incumbent: f32) -> Result<Self, String> {
        let value = f64::from(value);
        self.trust_observations += 1;
        if let Some(st) = self.success_tolerance_override
            && st != 3
        {
            let tol = 1.0e-3 * f64::from(incumbent).abs().max(1.0);
            let improved = Some(value > f64::from(incumbent) + tol);
            if self.adapt_tolerance(improved, st) {
                self.restart_bookkeeping()?;
                self.length = self.length_config.length_init;
            }
            return Ok(self);
        }
        self.trust
            .update_batch(
                &ndarray::ArrayView1::from(&[value]),
                self.trust_observations,
                f64::from(incumbent),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.restart_bookkeeping()?;
        }
        self.length = self.trust.length();
        Ok(self)
    }

    pub fn update_outcome(
        mut self,
        value: f32,
        incumbent: f32,
        outcome: TrustRegionOutcome,
    ) -> Result<Self, String> {
        self.trust_observations += 1;
        if let Some(st) = self.success_tolerance_override
            && st != 3
        {
            let improved = Some(outcome == TrustRegionOutcome::Success);
            if self.adapt_tolerance(improved, st) {
                self.restart_bookkeeping()?;
                self.length = self.length_config.length_init;
            }
            return Ok(self);
        }
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
            self.restart_bookkeeping()?;
        }
        self.length = self.trust.length();
        Ok(self)
    }
}
