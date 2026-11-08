use super::*;
pub use crate::coordinate::DomainFrame;

impl BayesOptimizer {
    /// Current observation counts in each isolated regional surrogate.
    ///
    /// Returns `None` when this optimizer has no multi-region local models.
    pub fn model_counts(&self) -> Option<Vec<usize>> {
        self.local_models.as_ref().map(|models| models.counts())
    }

    /// Number of configured local region slots, independent of observation
    /// count.
    pub fn num_regions(&self) -> usize {
        match &self.regions {
            TrustRegionState::Multi { state, .. } => state.num_regions(),
            _ => 1,
        }
    }

    /// Get current telemetry.
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Get bounds.
    pub fn bounds(&self) -> &Array2<f64> {
        self.domain.bounds()
    }

    /// Get number of dimensions.
    pub fn num_dim(&self) -> usize {
        self.domain.num_dim()
    }

    /// Get configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Get trust region state.
    pub fn trust_region(&self) -> &TrustRegionState {
        &self.regions
    }

    /// Get mutable trust region state.
    pub fn trust_mut(&mut self) -> &mut TrustRegionState {
        &mut self.regions
    }

    /// Trust region length (TuRBO, Morbo, or Multi-TR inner).
    pub fn tr_length(&self) -> f64 {
        self.regions.length()
    }

    /// Posterior mean at `x` in natural units, in natural `y` units.
    pub fn posterior_mu(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        crate::optimizer_mu::posterior_mu(
            self.domain.num_dim(),
            &self.domain.bounds().view(),
            self.posterior.surrogate(),
            x,
        )
    }

    /// Get observations in unit space.
    #[allow(dead_code)]
    pub(crate) fn unit_obs(&self) -> Option<Array2<f64>> {
        self.posterior.unit_obs()
    }

    /// Get observations in natural coordinates.
    pub fn x_obs(&self) -> Option<Array2<f64>> {
        let unit = self.posterior.unit_obs()?;
        Some(self.domain.from_unit(&unit.view()))
    }

    /// Get observation values.
    pub fn y_obs(&self) -> Option<Array2<f64>> {
        self.posterior.y_obs()
    }

    /// Get incumbent x in unit space.
    pub fn unit_incumbent(&self) -> Option<&Array1<f64>> {
        self.incumbent.unit_x()
    }

    /// Get the incumbent in natural coordinates.
    pub fn incumbent_x(&self) -> Option<Array1<f64>> {
        let unit = self.incumbent.unit_x()?;
        Some(self.domain.natural_point(unit))
    }

    /// Get incumbent y scalar.
    pub fn y_scalar(&self) -> Option<&Array1<f64>> {
        self.incumbent.y_scalar()
    }

    /// Current surrogate lengthscales (anisotropic trust-region sides), if any.
    pub fn lengthscales(&self) -> Option<Array1<f64>> {
        self.posterior.surrogate().and_then(|s| s.lengthscales())
    }

    /// Get init progress from warmup schedule.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        self.warmup.progress()
    }

    /// Get surrogate.
    pub fn surrogate(&self) -> Option<&(dyn crate::surrogate::Surrogate + Send + Sync)> {
        self.posterior.surrogate()
    }

    /// Whether this optimizer fits a surrogate (false for TuRBO-ZERO and
    /// LHD-only).
    pub fn has_surrogate(&self) -> bool {
        self.posterior.has_surrogate()
    }

    /// Get mutable surrogate.
    pub fn surrogate_mut(
        &mut self,
    ) -> Option<&mut (dyn crate::surrogate::Surrogate + Send + Sync)> {
        self.posterior.surrogate_mut()
    }

    /// Current number of stored observations.
    pub fn obs_count(&self) -> usize {
        self.posterior.obs_count()
    }

    pub fn update_incumbent(&mut self, rng: &mut dyn RngCore) -> Result<(), ENNError> {
        self.incumbent.update(
            &self.posterior,
            &self.regions,
            self.domain.num_dim(),
            self.config.noise_aware,
            rng,
        )
    }
}
