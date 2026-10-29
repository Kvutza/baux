use super::*;

/// Coordinate frame owning `[num_dim, 2]` natural bounds and unit-cube
/// transforms.
#[derive(Debug, Clone)]
pub struct DomainFrame {
    bounds: Array2<f64>,
    num_dim: usize,
}

impl DomainFrame {
    pub fn new(bounds: Array2<f64>) -> Result<Self, ENNError> {
        let num_dim = bounds.nrows();
        if bounds.ncols() != 2 {
            return Err(ENNError::InvalidShape {
                expected: vec![num_dim, 2],
                got: vec![num_dim, bounds.ncols()],
            });
        }
        Ok(Self { bounds, num_dim })
    }

    pub fn bounds(&self) -> &Array2<f64> {
        &self.bounds
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn to_unit(&self, x: &ArrayView2<f64>) -> Array2<f64> {
        crate::to_unit(x, &self.bounds.view())
    }

    pub fn from_unit(&self, unit: &ArrayView2<f64>) -> Array2<f64> {
        crate::from_unit(unit, &self.bounds.view())
    }

    pub fn natural_point(&self, unit: &Array1<f64>) -> Array1<f64> {
        let row = unit.clone().insert_axis(ndarray::Axis(0));
        let natural = crate::candidates::from_unit(&row.view(), &self.bounds.view());
        natural.row(0).to_owned()
    }

    pub fn unit_center(&self) -> Array1<f64> {
        Array1::from_elem(self.num_dim, 0.5)
    }
}

impl BayesOptimizer {
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

    /// Row-level observation access (unified posterior store).
    pub fn obs_access(&self) -> obs_access::ObsAccess<'_> {
        obs_access::ObsAccess::new(self)
    }

    /// Get surrogate.
    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        self.posterior.surrogate()
    }

    /// Whether this optimizer fits a surrogate (false for TuRBO-ZERO and
    /// LHD-only).
    pub fn has_surrogate(&self) -> bool {
        self.posterior.has_surrogate()
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

    /// Get mutable surrogate.
    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        self.posterior.surrogate_mut()
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
}
