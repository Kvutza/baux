//! Establish the resident optimizer's policies before observations and asks.
//!
//! Normal setup is `configure_enn`, optional `configure_controller`, then
//! `observe_initial`. Proposal program, noise sampler, history storage,
//! fitting, and radius control are separate choices. Unsupported combinations
//! are errors, not requests for a silent fallback to another optimizer.
//!
//! `latent_history` changes retained geometry/storage, not the number of model
//! parameters being optimized. Full rows/replay scratch are released in that
//! mode; supported FP16 innovation programs additionally retain sampled
//! parameter rows for neighbor distances. These samples are not full models.

use super::*;
use crate::params::ENNParams;

impl SearchState {
    /// Supply the initial scalar incumbent reward and estimate variance.
    /// Higher reward is better; pretraining passes negative mean NLL. Delegates
    /// initial-history/controller bookkeeping to `observe_objectives`; this
    /// method neither evaluates weights nor generates a proposal.
    pub fn observe_initial(&mut self, value: f32, variance: f32) -> Result<(), String> {
        self.observe_objectives(
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Set failure tolerance while idle and before any published ask.
    /// Updates the ordinary trust-region controller and any configured MORBO
    /// controller; it does not set reliability evidence thresholds or a radius.
    pub fn set_tolerance(&mut self, failures: usize) -> Result<(), String> {
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
        Ok(())
    }

    /// Install proposal, history, fitting, and query-neighbor policies.
    /// Requires idle implicit history with no observations and a valid ask.
    /// Sets warm-up count to configured k, prepares the fitter, and compiles
    /// the pool. Latent mode releases full history storage; compatible FP16
    /// Gaussian and Rademacher programs then allocate sampled parameter
    /// geometry. Pair rotation and threshold programs have explicit
    /// compatibility guards. Fitting variance scales and query-local
    /// neighbor adaptation are separate flags. Configuration can
    /// allocate/compile and is not rollback-atomic on failure; do not reuse
    /// a partially configured state after an error.
    pub fn configure_enn(
        &mut self,
        config: crate::config::ResidentEnnConfig,
    ) -> Result<(), String> {
        self.check_idle()?;
        check_ask(config.ask)?;
        self.configure_pool(config.pool)?;
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
        Ok(())
    }

    /// Last fitted `(k, epistemic_scale, aleatoric_scale, y_scale)`, if
    /// available. Reading this telemetry does not fit parameters or select
    /// a local k.
    pub fn fitted_enn(&self) -> Option<(usize, f32, f32, f32)> {
        self.fitted_enn
    }

    /// Current count of recorded history slots.
    pub fn history(&self) -> usize {
        self.history
    }
}
