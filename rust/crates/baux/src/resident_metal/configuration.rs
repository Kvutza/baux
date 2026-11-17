//! Establish the resident optimizer's policies before observations and asks.

use super::*;
use crate::params::ENNParams;

impl SearchState {
    pub fn observe_initial(self, value: f32, variance: f32) -> Result<Self, String> {
        self.observe_objectives(
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Compute Host-aligned TuRBO failure tolerance `ceil(max(4/arms,
    /// D/arms))`.
    pub fn failure_tolerance(dimensions: usize, arms: usize) -> usize {
        baux::semantics::optimizer::turbo_tolerance(dimensions.max(1) as f64, arms.max(1)) as usize
    }

    pub fn set_tolerance(mut self, failures: usize) -> Result<Self, String> {
        self.check_idle()?;
        if self.started {
            return Err("Set Metal BF16 failure tolerance before any asks".into());
        }
        self.trust
            .set_tolerance(failures)
            .map_err(|error| error.to_string())?;
        if let Some(morbo) = self
            .objective_selector
            .as_mut()
            .and_then(|selector| selector.morbo.as_mut())
        {
            morbo.set_tolerance(failures)?;
        }
        Ok(self)
    }

    pub fn set_successes(mut self, successes: usize) -> Result<Self, String> {
        self.check_idle()?;
        if self.started {
            return Err("Set Metal BF16 success tolerance before any asks".into());
        }
        let s = i32::try_from(successes)
            .ok()
            .filter(|&v| v > 0)
            .ok_or("success tolerance must be in 1..=i32::MAX")?;
        self.success_tolerance_override = Some(s);
        Ok(self)
    }

    pub fn fix_radius(mut self, fixed: bool) -> Result<Self, String> {
        self.check_idle()?;
        self.adapt_controller = !fixed;
        Ok(self)
    }

    pub fn configure_enn(
        mut self,
        config: crate::config::ResidentEnnConfig,
    ) -> Result<Self, String> {
        self.check_idle()?;
        check_ask(config.ask)?;
        self = self.configure_pool(config.pool)?;
        if !self.implicit_history || self.history != 0 {
            return Err("Configure implicit ENN before its first observation".into());
        }
        if config.ask.surrogate.neighbors > MAX_HISTORY {
            return Err(format!(
                "Implicit ENN k={} exceeds logical history capacity {MAX_HISTORY}",
                config.ask.surrogate.neighbors
            ));
        }
        let k = i32::try_from(config.ask.surrogate.neighbors)
            .map_err(|_| "Implicit ENN k does not fit i32")?;
        let mut fitter = ENNFitter::new(k, true);
        fitter.set_params(
            ENNParams::new(
                k,
                f64::from(config.ask.surrogate.epistemic_scale),
                f64::from(config.ask.surrogate.aleatoric_scale),
            )
            .map_err(|error| error.to_string())?,
        );
        self.initial_observations = config.ask.surrogate.neighbors;
        self.fit_candidates = config.num_candidates;
        self.fit_samples = config.num_samples;
        self.fit_params = config.fit_params;
        self.adapt_neighbors = config.adapt_neighbors;
        self.fit_seed = config.ask.acquisition.seed;
        self.distance_metric = config.distance_metric;
        self.local_scale_neighbors = config.local_scale_neighbors;
        self.fitter = Some(fitter);
        Ok(self)
    }

    pub fn fitted_enn(&self) -> Option<(usize, f32, f32, f32)> {
        self.fitted_enn
    }

    pub fn history(&self) -> usize {
        self.history
    }
}
