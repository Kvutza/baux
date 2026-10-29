//! ENN parameter and configuration data structures.

use crate::error::ENNError;
use crate::y_bounds::OutputTransform;
use ndarray::{Array2, Array3, Array4, ArrayD, Ix2, Ix3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

/// Errors that can occur when creating ENN parameters.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParamsError {
    /// Invalid number of neighbors.
    #[error("k_neighbors must be > 0, got {0}")]
    InvalidK(i32),
    /// Invalid epistemic variance scale.
    #[error("epistemic_scale must be >= 0 and finite, got {0}")]
    InvalidEpistemicVariance(f64),
    /// Invalid aleatoric variance scale.
    #[error("aleatoric_scale must be >= 0 and finite, got {0}")]
    InvalidAleatoricVariance(f64),
}

/// Parameters for Epistemic Nearest Neighbors computation.
///
/// These parameters control the behavior of the ENN posterior computation,
/// including the number of neighbors and variance scaling factors.
///
/// # Fields
///
/// * `k_neighbors` - Number of nearest neighbors to use (must be > 0)
/// * `epistemic_scale` - Scale factor for epistemic (model) uncertainty
/// * `aleatoric_scale` - Scale factor for aleatoric (observation) noise
///
/// # Example
///
/// ```
/// use baux::params::ENNParams;
///
/// let params = ENNParams::new(5, 1.0, 0.1).unwrap();
/// assert_eq!(params.k_neighbors, 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ENNParams {
    /// Number of nearest neighbors to use.
    pub k_neighbors: i32,
    /// Scale factor for epistemic uncertainty.
    pub epistemic_scale: f64,
    /// Scale factor for aleatoric noise.
    pub aleatoric_scale: f64,
}

impl ENNParams {
    /// Create a new ENNParams instance with validation.
    ///
    /// # Arguments
    ///
    /// * `k_neighbors` - Number of neighbors (must be > 0)
    /// * `epistemic_scale` - Epistemic variance scale (must be >= 0 and finite)
    /// * `aleatoric_scale` - Aleatoric variance scale (must be >= 0 and finite)
    ///
    /// # Errors
    ///
    /// Returns `ParamsError` if any parameter is invalid.
    pub fn new(
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<Self, ParamsError> {
        // Validate k
        if k_neighbors <= 0 {
            return Err(ParamsError::InvalidK(k_neighbors));
        }

        // Validate epistemic variance
        if !epistemic_scale.is_finite() || epistemic_scale < 0.0 {
            return Err(ParamsError::InvalidEpistemicVariance(epistemic_scale));
        }

        // Validate aleatoric variance
        if !aleatoric_scale.is_finite() || aleatoric_scale < 0.0 {
            return Err(ParamsError::InvalidAleatoricVariance(aleatoric_scale));
        }

        Ok(Self {
            k_neighbors,
            epistemic_scale,
            aleatoric_scale,
        })
    }
}

/// Flags controlling posterior computation behavior.
///
/// These flags allow fine-grained control over how the ENN posterior
/// is computed.
///
/// # Fields
///
/// * `exclude_nearest` - Whether to exclude the nearest neighbor from
///   computation
/// * `observation_noise` - Whether to include observation noise in the
///   uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PosteriorFlags {
    /// Exclude the nearest neighbor from computation.
    pub exclude_nearest: bool,
    /// Include observation noise in uncertainty computation.
    pub observation_noise: bool,
    /// Break distance ties by lower train index (Exact driver neighbor lookup).
    pub tie_neighbors: bool,
}

impl PosteriorFlags {
    /// Create new PosteriorFlags with default values (all false except
    /// tie_neighbors).
    pub fn new() -> Self {
        Self {
            tie_neighbors: true,
            ..Self::default()
        }
    }

    /// Set exclude_nearest flag.
    pub fn exclude_nearest(mut self, value: bool) -> Self {
        self.exclude_nearest = value;
        self
    }

    /// Set observation_noise flag.
    pub fn observation_noise(mut self, value: bool) -> Self {
        self.observation_noise = value;
        self
    }

    /// Set tie_neighbors flag.
    pub fn tie_neighbors(mut self, value: bool) -> Self {
        self.tie_neighbors = value;
        self
    }
}

/// Normal distribution result from ENN posterior computation.
///
/// Represents the posterior predictive distribution as a Gaussian
/// with mean and standard error.
///
/// # Fields
///
/// * `mu` - Predictive mean for each query point
/// * `se` - Predictive standard error for each query point
/// * `se_epi` - Epistemic (model) standard error component
/// * `se_ale` - Aleatoric (observation) standard error component
/// * `idx` - Optional neighbor indices used for each prediction
#[derive(Debug, Clone, PartialEq)]
pub struct ENNNormal {
    /// Predictive means.
    pub mu: ArrayD<f64>,
    /// Predictive standard errors.
    pub se: ArrayD<f64>,
    /// Epistemic standard error component.
    pub se_epi: ArrayD<f64>,
    /// Aleatoric standard error component.
    pub se_ale: ArrayD<f64>,
    /// Optional neighbor indices, shape `(n_query, k)`.
    pub idx: Option<Array2<i64>>,
    /// Optional per-metric output bounds `(num_metrics, 2)`.
    pub y_bounds: Option<Array2<f64>>,
}

impl ENNNormal {
    /// Create a new ENNNormal instance.
    pub fn new(
        mu: ArrayD<f64>,
        se: ArrayD<f64>,
        se_epi: ArrayD<f64>,
        se_ale: ArrayD<f64>,
        idx: Option<Array2<i64>>,
    ) -> Self {
        Self {
            mu,
            se,
            se_epi,
            se_ale,
            idx,
            y_bounds: None,
        }
    }

    /// Attach output-space bounds for quantile and sampling transforms.
    pub fn with_bounds(mut self, bounds: Array2<f64>) -> Self {
        self.y_bounds = Some(bounds);
        self
    }

    /// Get the number of query points.
    pub fn len(&self) -> usize {
        self.mu.len()
    }

    /// Check if there are no query points.
    pub fn is_empty(&self) -> bool {
        self.mu.is_empty()
    }

    fn view_3d(
        &self,
    ) -> Result<
        (
            ndarray::ArrayView3<'_, f64>,
            ndarray::ArrayView3<'_, f64>,
            bool,
        ),
        ENNError,
    > {
        if self.mu.shape() != self.se.shape() {
            return Err(ENNError::InvalidShape {
                expected: self.mu.shape().to_vec(),
                got: self.se.shape().to_vec(),
            });
        }
        match self.mu.ndim() {
            2 => {
                let mu2 = self.mu.view().into_dimensionality::<Ix2>().unwrap();
                let se2 = self.se.view().into_dimensionality::<Ix2>().unwrap();
                let (n, m) = mu2.dim();
                Ok((
                    mu2.into_shape_with_order((1, n, m)).unwrap(),
                    se2.into_shape_with_order((1, n, m)).unwrap(),
                    true,
                ))
            }
            3 => Ok((
                self.mu.view().into_dimensionality::<Ix3>().unwrap(),
                self.se.view().into_dimensionality::<Ix3>().unwrap(),
                false,
            )),
            _ => Err(ENNError::InvalidShape {
                expected: vec![0, 0],
                got: self.mu.shape().to_vec(),
            }),
        }
    }

    fn latent_grid(
        mu3: ndarray::ArrayView3<'_, f64>,
        se3: ndarray::ArrayView3<'_, f64>,
        bounds: &dyn OutputTransform,
    ) -> Result<Array3<(f64, f64)>, ENNError> {
        let (p_dim, n, m_dim) = mu3.dim();
        if bounds.metrics() != m_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![m_dim, 2],
                got: vec![bounds.metrics(), 2],
            });
        }
        let mut grid = Array3::from_elem((p_dim, n, m_dim), (0.0, 0.0));
        for p in 0..p_dim {
            for i in 0..n {
                for m in 0..m_dim {
                    let z_mu = bounds.warp_val(m, mu3[[p, i, m]])?;
                    let jac = bounds.jac_val(m, z_mu).max(1e-30);
                    grid[[p, i, m]] = (z_mu, se3[[p, i, m]] / jac);
                }
            }
        }
        Ok(grid)
    }

    fn transform_z(&self, z: f64) -> Result<ArrayD<f64>, ENNError> {
        let (mu3, se3, squeeze_p) = self.view_3d()?;
        let (p_dim, n, m_dim) = mu3.dim();
        let mut out = Array3::zeros((p_dim, n, m_dim));
        if let Some(bounds) = self.y_bounds.as_ref().filter(|b| !b.is_identity()) {
            let grid = Self::latent_grid(mu3, se3, bounds)?;
            for ((p, i, m), &(z_mu, sigma_z)) in grid.indexed_iter() {
                out[[p, i, m]] = bounds.inv_val(m, z_mu + z * sigma_z);
            }
        } else {
            for ((p, i, m), &mu_val) in mu3.indexed_iter() {
                out[[p, i, m]] = mu_val + z * se3[[p, i, m]];
            }
        }
        if squeeze_p {
            Ok(out.index_axis_move(ndarray::Axis(0), 0).into_dyn())
        } else {
            Ok(out.into_dyn())
        }
    }

    /// Predictive quantile at probability $p \in (0, 1)$, respecting output
    /// bounds when present.
    pub fn quantile(&self, p: f64) -> Result<ArrayD<f64>, ENNError> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(ENNError::InvalidParameter(
                "quantile p must be in (0, 1)".to_string(),
            ));
        }
        self.transform_z(crate::ndtri::ndtri(p))
    }

    /// Symmetric two-sided predictive confidence interval `(lower, upper)` at
    /// `level` $\in (0, 1)$.
    pub fn confidence_interval(&self, level: f64) -> Result<(ArrayD<f64>, ArrayD<f64>), ENNError> {
        let z = crate::ndtri::z_crit(level)?;
        Ok((self.transform_z(-z)?, self.transform_z(z)?))
    }

    /// Draw `num_samples` independent normal samples per query point and
    /// metric.
    pub fn sample(
        &self,
        num_samples: usize,
        seed: u64,
        clip: Option<f64>,
    ) -> Result<ArrayD<f64>, ENNError> {
        if num_samples == 0 {
            return Err(ENNError::InvalidParameter(
                "num_samples must be > 0".to_string(),
            ));
        }
        if let Some(c) = clip {
            if !c.is_finite() || c <= 0.0 {
                return Err(ENNError::InvalidParameter(
                    "clip must be positive and finite".to_string(),
                ));
            }
        }
        let (mu3, se3, squeeze_p) = self.view_3d()?;
        let (p_dim, n, m_dim) = mu3.dim();
        let grid = match self.y_bounds.as_ref().filter(|b| !b.is_identity()) {
            Some(bounds) => Some(Self::latent_grid(mu3, se3, bounds)?),
            None => None,
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Array4::zeros((num_samples, p_dim, n, m_dim));
        for s in 0..num_samples {
            for p in 0..p_dim {
                for i in 0..n {
                    for m in 0..m_dim {
                        let mut eps: f64 = StandardNormal.sample(&mut rng);
                        if let Some(c) = clip {
                            eps = eps.clamp(-c, c);
                        }
                        out[[s, p, i, m]] = match (&grid, &self.y_bounds) {
                            (Some(g), Some(b)) => {
                                let (z_mu, sigma_z) = g[[p, i, m]];
                                b.inv_val(m, z_mu + sigma_z * eps)
                            }
                            _ => mu3[[p, i, m]] + se3[[p, i, m]] * eps,
                        };
                    }
                }
            }
        }
        if squeeze_p {
            Ok(out.index_axis_move(ndarray::Axis(1), 0).into_dyn())
        } else {
            Ok(out.into_dyn())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, array};

    #[test]
    fn test_ennparamsvalid() {
        let params = ENNParams::new(5, 1.0, 0.1).unwrap();
        assert_eq!(params.k_neighbors, 5);
        assert_eq!(params.epistemic_scale, 1.0);
        assert_eq!(params.aleatoric_scale, 0.1);
    }

    #[test]
    fn test_ennparamsinvalidk() {
        assert!(matches!(
            ENNParams::new(0, 1.0, 0.1),
            Err(ParamsError::InvalidK(0))
        ));
        assert!(matches!(
            ENNParams::new(-1, 1.0, 0.1),
            Err(ParamsError::InvalidK(-1))
        ));
    }

    #[test]
    fn test_003() {
        assert!(matches!(
            ENNParams::new(5, -1.0, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(-1.0))
        ));
        assert!(matches!(
            ENNParams::new(5, f64::NAN, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(_))
        ));
        assert!(matches!(
            ENNParams::new(5, f64::INFINITY, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(_))
        ));
    }

    #[test]
    fn test_004() {
        assert!(matches!(
            ENNParams::new(5, 1.0, -0.1),
            Err(ParamsError::InvalidAleatoricVariance(-0.1))
        ));
    }

    #[test]
    fn test_005() {
        let flags = PosteriorFlags::new();
        assert!(!flags.exclude_nearest);
        assert!(!flags.observation_noise);
        assert!(flags.tie_neighbors);
    }

    #[test]
    fn test_006() {
        let flags = PosteriorFlags::new()
            .exclude_nearest(true)
            .observation_noise(true);
        assert!(flags.exclude_nearest);
        assert!(flags.observation_noise);
    }

    #[test]
    fn test_ennnormal() {
        let mu = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let se = array![[0.1, 0.2], [0.3, 0.4]].into_dyn();
        let se_epi = se.clone();
        let se_ale = array![[0.0, 0.0], [0.0, 0.0]].into_dyn();
        let normal = ENNNormal::new(mu.clone(), se.clone(), se_epi, se_ale, None);

        assert_eq!(normal.len(), 4);
        assert!(!normal.is_empty());
        assert_eq!(normal.mu, mu);
        assert_eq!(normal.se, se);
    }

    #[test]
    fn test_ennnormalwithidx() {
        let mu = array![[1.0], [2.0]].into_dyn();
        let se = array![[0.1], [0.2]].into_dyn();
        let se_epi = se.clone();
        let se_ale = array![[0.0], [0.0]].into_dyn();
        let idx = Some(array![[0, 1], [1, 2]]);
        let normal = ENNNormal::new(mu, se, se_epi, se_ale, idx.clone());

        assert_eq!(normal.idx, idx);
    }

    #[test]
    fn test_ennnormalempty() {
        let mu = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let se = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let se_epi = se.clone();
        let se_ale = se.clone();
        let normal = ENNNormal::new(mu, se, se_epi, se_ale, None);

        assert!(normal.is_empty());
        assert_eq!(normal.len(), 0);
    }

    #[test]
    fn normal_quantiles() {
        let mu = array![[0.0], [10.0]].into_dyn();
        let se = array![[1.0], [2.0]].into_dyn();
        let post = ENNNormal::new(mu, se.clone(), se, ArrayD::zeros(IxDyn(&[2, 1])), None);
        let (lo, hi) = post.confidence_interval(0.95).unwrap();
        let z95 = 1.959_963_984_540_054;
        assert!((lo[[0, 0]] + z95).abs() < 1e-12);
        assert!((hi[[0, 0]] - z95).abs() < 1e-12);
        assert!((lo[[1, 0]] - (10.0 - 2.0 * z95)).abs() < 1e-12);
        assert!((hi[[1, 0]] - (10.0 + 2.0 * z95)).abs() < 1e-12);
        let median = post.quantile(0.5).unwrap();
        assert!((median[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((median[[1, 0]] - 10.0).abs() < 1e-12);
        let draws = post.sample(32, 7, Some(2.0)).unwrap();
        assert_eq!(draws.shape(), &[32, 2, 1]);
        for s in 0..32 {
            assert!(draws[[s, 0, 0]].abs() <= 2.0 + 1e-12);
        }
    }

    #[test]
    fn bounded_normal() {
        let bounds = array![[0.0, 1.0]];
        let mu = array![[0.05], [0.95]].into_dyn();
        let se = array![[0.4], [0.4]].into_dyn();
        let post = ENNNormal::new(mu, se.clone(), se, ArrayD::zeros(IxDyn(&[2, 1])), None)
            .with_bounds(bounds);
        let (lo, hi) = post.confidence_interval(0.99).unwrap();
        for i in 0..2 {
            assert!(lo[[i, 0]] > 0.0 && lo[[i, 0]] < 1.0);
            assert!(hi[[i, 0]] > 0.0 && hi[[i, 0]] < 1.0);
            assert!(lo[[i, 0]] < post.mu[[i, 0]] && post.mu[[i, 0]] < hi[[i, 0]]);
        }
        let draws = post.sample(64, 19, None).unwrap();
        assert_eq!(draws.shape(), &[64, 2, 1]);
        assert!(draws.iter().all(|&v| v > 0.0 && v < 1.0));
    }
}
