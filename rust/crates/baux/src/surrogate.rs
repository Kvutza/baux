//! Surrogate models for optimization.

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;
use rand::SeedableRng;

use std::path::PathBuf;

use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::fitter::ENNFitter;
use crate::index::IndexDriver;
pub use crate::layout::MetricLearning;
use crate::model::ENN;
use crate::params::{ENNParams, PosteriorFlags};
pub use crate::surrogate_api::{BoxedSurrogate, Surrogate, SurrogatePrediction};
use crate::traits::PosteriorComputation;

#[derive(Debug, Clone)]
pub struct ENNSurrogateConfig {
    pub k: i32,
    pub scale_x: bool,
    pub num_candidates: usize,
    pub num_samples: usize,
    pub infer_aleatoric_variance: bool,
    pub index_driver: IndexDriver,
    pub storage: EnnStorage,
    pub work_dir: Option<PathBuf>,
    pub y_bounds: Option<Array2<f64>>,
    pub metric_learning: MetricLearning,
    pub tied_dims: Vec<Vec<usize>>,
    pub unscaled_dims: Vec<usize>,
    pub geometry: crate::semantics::neighbor::DistanceMetric,
    pub stiefel: Option<crate::semantics::stiefel::Stiefel>,
    pub affine: Option<crate::semantics::affine::Affine>,
}

impl Default for ENNSurrogateConfig {
    fn default() -> Self {
        Self {
            k: 10,
            scale_x: false,
            num_candidates: 30,
            num_samples: 10,
            infer_aleatoric_variance: true,
            index_driver: IndexDriver::Exact,
            storage: EnnStorage::InMemory,
            work_dir: None,
            y_bounds: None,
            metric_learning: MetricLearning::None,
            tied_dims: Vec::new(),
            unscaled_dims: Vec::new(),
            geometry: crate::semantics::neighbor::DistanceMetric::Euclidean,
            stiefel: None,
            affine: None,
        }
    }
}

impl ENNSurrogateConfig {
    pub fn validate(&self, num_dim: usize) -> Result<(), ENNError> {
        crate::metric_weights::validate_tied(&self.tied_dims, num_dim)?;
        if self.unscaled_dims.iter().any(|&d| d >= num_dim) {
            return Err(ENNError::InvalidParameter(
                "unscaled_dims contains out-of-range dimension".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct ENNSurrogate {
    config: ENNSurrogateConfig,
    model: Option<ENN>,
    params: Option<ENNParams>,
    fitter: Option<ENNFitter>,
}

impl ENNSurrogate {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config,
            model: None,
            params: None,
            fitter: None,
        }
    }

    pub fn model(&self) -> Option<&ENN> {
        self.model.as_ref()
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.params.as_ref()
    }

    fn construct_model(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<ENN, ENNError> {
        let mut model = ENN::new_options(
            x.to_owned(),
            y.to_owned(),
            yvar.map(|v| v.to_owned()),
            crate::ModelOptions {
                scale_x: self.config.scale_x,
                driver: self.config.index_driver,
                storage: self.config.storage,
                work_dir: self.config.work_dir.clone(),
                y_bounds: self.config.y_bounds.clone(),
            },
        )?;
        if self.config.metric_learning == MetricLearning::Auto {
            let seed = rng.next_u64();
            model.enable_metric(self.config.tied_dims.clone(), seed)?;
        }
        if !self.config.tied_dims.is_empty() && model.tied_groups().is_empty() {
            model.set_tied(self.config.tied_dims.clone())?;
        }
        if !self.config.unscaled_dims.is_empty()
            && (self.config.scale_x || self.config.metric_learning == MetricLearning::Auto)
        {
            model.set_unscaled(self.config.unscaled_dims.clone())?;
        }
        if self.config.geometry != crate::semantics::neighbor::DistanceMetric::Euclidean {
            model = model.with_geometry(self.config.geometry);
        }
        if let Some(st) = self.config.stiefel.clone() {
            model = model.with_stiefel(st)?;
        }
        if let Some(aff) = self.config.affine.clone() {
            model = model.with_affine(aff)?;
        }
        Ok(model)
    }

    fn run_fitter(&mut self, rng: &mut rand::rngs::StdRng) -> Result<(), ENNError> {
        if self.config.num_candidates == 0 || self.config.num_samples == 0 {
            if self.params.is_none() {
                self.params = Some(
                    ENNParams::new(self.config.k, 1.0, 0.0)
                        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?,
                );
            }
            return Ok(());
        }
        if self.fitter.is_none() {
            let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance)
                .with_bounds(self.config.y_bounds.clone());
            if let Some(p) = self.params {
                fitter.set_params(p);
            }
            self.fitter = Some(fitter);
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let fitter = self.fitter.as_mut().expect("fitter");
        let p = fitter.ask(
            model,
            self.config.num_candidates,
            self.config.num_samples,
            self.params.as_ref(),
            rng,
        )?;
        self.params = Some(p);
        Ok(())
    }
}

impl Surrogate for ENNSurrogate {
    fn clear_observations(&mut self) -> Result<(), ENNError> {
        let owned_store = if let Some(model) = &self.model {
            model.backend.wait_flush()?;
            matches!(self.config.storage, EnnStorage::Disk)
                .then(|| self.config.work_dir.clone().or_else(EnnStorage::work_env))
                .flatten()
        } else {
            None
        };
        self.model = None;
        self.params = None;
        self.fitter = None;
        if let Some(work_dir) = owned_store {
            baux::neighbor_index::remove_store(&work_dir)
                .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        }
        Ok(())
    }

    fn fitted_metrics(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.num_metrics())
    }

    fn observation_count(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.len())
    }

    fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        model.rows().row_x(idx)
    }

    fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        model.natural_y(idx)
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };
        let n = model.len();
        if n == 0 {
            return Ok(None);
        }
        // Gather natural y row-by-row without materializing full x (Θ(N·D)
        // RAM).
        let mut y = Array2::zeros((n, model.num_metrics()));
        for i in 0..n {
            y.row_mut(i).assign(&model.natural_y(i)?);
        }
        Ok(Some(y))
    }

    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError> {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };
        let n = model.len();
        if n == 0 {
            return Ok(None);
        }
        let mut x = Array2::zeros((n, model.num_dim()));
        for i in 0..n {
            x.row_mut(i).assign(&model.rows().row_x(i)?);
        }
        Ok(Some(x))
    }

    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        let mut local_rng = rand::rngs::StdRng::from_seed(seed_bytes);

        let model = self.construct_model(x, y, yvar, &mut local_rng)?;

        let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance)
            .with_bounds(self.config.y_bounds.clone());
        fitter.tell(x, y, yvar)?;
        if let Some(p) = self.params {
            fitter.set_params(p);
        }
        let p = fitter.ask(
            &model,
            self.config.num_candidates,
            self.config.num_samples,
            self.params.as_ref(),
            &mut local_rng,
        )?;
        self.params = Some(p);
        self.model = Some(model);
        self.fitter = Some(fitter);

        Ok(())
    }

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        const BULK_ROWS: usize = 4_096;
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        let mut local_rng = rand::rngs::StdRng::from_seed(seed_bytes);
        let is_disk = self.config.index_driver == IndexDriver::BpAnnDisk;
        let bulk_disk = is_disk && x_new.nrows() >= BULK_ROWS;
        let skip_fit = if let Some(model) = &mut self.model {
            model.add(x_new, y_new, yvar_new)?;
            if let Some(fitter) = self.fitter.as_mut() {
                fitter.tell(x_new, y_new, yvar_new)?;
            }
            bulk_disk || (is_disk && self.params.is_some())
        } else {
            let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance)
                .with_bounds(self.config.y_bounds.clone());
            fitter.tell(x_new, y_new, yvar_new)?;
            self.model = Some(self.construct_model(x_new, y_new, yvar_new, &mut local_rng)?);
            self.fitter = Some(fitter);
            bulk_disk
        };
        if !skip_fit || bulk_disk {
            if let Some(model) = &self.model {
                model.ensure_sync()?;
            }
        }
        if !skip_fit {
            self.run_fitter(&mut local_rng)?;
        }
        if (!skip_fit || bulk_disk) && is_disk {
            if let Some(model) = &self.model {
                model.index_access().release_pages()?;
            }
        }
        Ok(())
    }

    fn schedule_flush(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            model.backend.schedule_flush()
        } else {
            Ok(())
        }
    }

    fn wait_flush(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            // Join in-flight soft sync only. Do not ensure_sync here:
            // tell() calls this on every observation, and a forced drain turns
            // --tell-all (one-row tells) into O(N) soft-syncs.
            model.backend.wait_flush()
        } else {
            Ok(())
        }
    }

    fn release_pages(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            model.index_access().release_pages()
        } else {
            Ok(())
        }
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let params = match self.params.as_ref() {
            Some(p) => *p,
            None => {
                // Bulk disk tells may skip neighbor fit; use default params
                // until a small tell refreshes hyperparameters.
                ENNParams::new(self.config.k, 1.0, 0.0).map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                })?
            }
        };

        let flags = PosteriorFlags::new().tie_neighbors(false);
        let posterior = model.posterior(x, &params, &flags)?;

        // Convert from dynamic dimension to fixed 2D
        let mu = posterior
            .mu
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;
        let se = posterior
            .se
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;

        Ok(SurrogatePrediction { mu, se })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let params = self
            .params
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;

        // Use deterministic seeds based on current state
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let base_seed = u64::from_le_bytes(seed_bytes) as i64;
        let function_seeds: Vec<i64> = (0..num_samples as i64).map(|i| base_seed + i).collect();

        let (draws, _) = model.posterior_draw(x, params, &function_seeds, &Default::default())?;

        Ok(draws)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        self.model.as_ref()?.trust_sides().map(Array1::from)
    }

    fn y_bounds(&self) -> Option<&Array2<f64>> {
        self.model
            .as_ref()
            .filter(|m| m.bounded_outputs())
            .map(ENN::y_bounds)
            .or(self.config.y_bounds.as_ref())
    }

    fn naturalize_y(&self, z: &ArrayView2<f64>) -> Array2<f64> {
        match self.y_bounds() {
            Some(bounds) => crate::y_bounds::inv_y(*z, bounds),
            None => z.to_owned(),
        }
    }

    fn warp_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        match self.y_bounds() {
            Some(bounds) => crate::y_bounds::warp_y(*y, bounds),
            None => Ok(y.to_owned()),
        }
    }

    fn clone_box(&self) -> Result<BoxedSurrogate, ENNError> {
        Ok(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_001() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 5,
            num_samples: 3,
            ..Default::default()
        };
        let mut surrogate = ENNSurrogate::new(config);

        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [1.0], [2.0]];

        let mut rng = StdRng::seed_from_u64(42);
        surrogate.fit(&x.view(), &y.view(), None, &mut rng).unwrap();

        // Check model is fitted
        assert!(surrogate.model().is_some());
        assert!(surrogate.params().is_some());

        // Predict
        let x_query = array![[0.5, 0.5]];
        let pred = surrogate.predict(&x_query.view()).unwrap();
        assert_eq!(pred.mu.shape(), &[1, 1]);
        assert!(pred.mu[[0, 0]].is_finite());
    }

    /// Regression: incremental `fit` must not reuse stale `train_yvar` when the
    /// caller updates observation noise on prefix rows (same `x`/`y`, new
    /// `yvar` on old rows).
    #[test]
    fn regression_refit() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        };
        let x0 = array![[0.0, 0.0], [1.0, 0.0]];
        let y0 = array![[0.0], [1.0]];
        let yvar0 = array![[1.0], [2.0]];
        let x1 = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let y1 = array![[0.0], [1.0], [5.0]];
        let yvar1 = array![[1.0e6], [2.0], [1.0]];

        let mut rng_a = StdRng::seed_from_u64(11);
        let mut sur_inc = ENNSurrogate::new(config.clone());
        sur_inc
            .fit(&x0.view(), &y0.view(), Some(&yvar0.view()), &mut rng_a)
            .unwrap();
        sur_inc
            .fit(&x1.view(), &y1.view(), Some(&yvar1.view()), &mut rng_a)
            .unwrap();
        let v_inc = sur_inc
            .model()
            .unwrap()
            .rows()
            .row_yvar(0)
            .unwrap()
            .unwrap()[[0]];

        let mut rng_b = StdRng::seed_from_u64(11);
        let mut sur_full = ENNSurrogate::new(config);
        sur_full
            .fit(&x1.view(), &y1.view(), Some(&yvar1.view()), &mut rng_b)
            .unwrap();
        let v_full = sur_full
            .model()
            .unwrap()
            .rows()
            .row_yvar(0)
            .unwrap()
            .unwrap()[[0]];

        assert!(
            (v_inc - v_full).abs() < 1e-9,
            "train_yvar row0 incremental={v_inc} full_refit={v_full} (prefix yvar must refresh)"
        );
    }

    #[test]
    fn y_append() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        };
        let x0 = array![[0.0, 0.0], [1.0, 0.0]];
        let y0 = array![[0.0], [1.0]];
        let x1 = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let y1 = array![[0.0], [1.0], [f64::NAN]];

        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(42);
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();
        let result = sur.fit(&x1.view(), &y1.view(), None, &mut rng);
        assert!(
            result.is_err(),
            "non-finite y on incremental append must be rejected (use tell)"
        );
    }

    #[test]
    fn test_004() {
        let pred = SurrogatePrediction {
            mu: array![[1.0], [2.0]],
            se: array![[0.1], [0.2]],
        };
        let cloned = pred.clone();
        assert_eq!(cloned.mu.shape(), &[2, 1]);
        assert_eq!(cloned.se.shape(), &[2, 1]);
        assert_eq!(cloned.mu[[1, 0]], 2.0);

        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(42);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y = array![[0.0], [1.0], [2.0]];
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        let mut boxed = sur.clone_box().unwrap();
        boxed
            .fit_append(
                &array![[1.0, 1.0]].view(),
                &array![[10.0]].view(),
                None,
                &mut rng,
            )
            .unwrap();
        assert_eq!(sur.observation_count(), Some(3));
        assert_eq!(boxed.observation_count(), Some(4));
        assert!(ENNSurrogateConfig::default().k >= 1);
    }

    #[test]
    fn fit_search() {
        use crate::backend::EnnStorage;
        use crate::index::IndexDriver;
        use ndarray::Array2;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            scale_x: false,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(7);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5]];
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();

        let x1 = Array2::from_shape_fn((4_096, 2), |(i, j)| (i + j) as f64 * 0.001);
        let y1 = Array2::from_shape_fn((4_096, 1), |(i, _)| (i as f64) * 0.01);
        sur.fit_append(&x1.view(), &y1.view(), None, &mut rng)
            .unwrap();

        let model = sur.model().expect("model");
        // Pending must be drained: otherwise disk search brute-forces
        // Θ(pending). Soft sync writes indexed_rows.bin; metadata may
        // lag until hard persist.
        let indexed = std::fs::read(dir.path().join("indexed_rows.bin"))
            .ok()
            .and_then(|b| {
                if b.len() >= 8 {
                    Some(u64::from_le_bytes(b[..8].try_into().ok()?) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        assert_eq!(
            indexed,
            model.len(),
            "indexed_rows.bin must match num_obs after fit_append sync-before-fit"
        );
    }

    #[test]
    fn disk_restart() {
        use crate::backend::EnnStorage;
        use crate::index::IndexDriver;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let mut surrogate = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(5);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5], [2.0]];
        surrogate
            .fit(&x0.view(), &y0.view(), None, &mut rng)
            .unwrap();
        let other = dir.path().join("not_bpann.txt");
        std::fs::write(&other, "keep").unwrap();

        surrogate.clear_observations().unwrap();
        assert_eq!(surrogate.observation_count(), None);

        let x1 = array![[0.2, 0.2], [0.8, 0.3], [0.4, 0.9]];
        let y1 = array![[3.0], [4.0], [5.0]];
        surrogate
            .fit(&x1.view(), &y1.view(), None, &mut rng)
            .unwrap();
        assert_eq!(surrogate.observation_count(), Some(3));
        assert_eq!(surrogate.observations_y().unwrap().unwrap(), y1);
        assert_eq!(std::fs::read_to_string(other).unwrap(), "keep");
    }

    #[test]
    fn fit_exist() {
        use crate::backend::EnnStorage;
        use crate::index::IndexDriver;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            scale_x: false,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(11);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5]];
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();
        assert!(sur.params().is_some(), "initial fit must set params");

        // One-row tells must not force soft-sync (tell-all path).
        let before = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        for i in 0..20 {
            let x = array![[i as f64 * 0.01, 0.1]];
            let y = array![[i as f64 * 0.1]];
            sur.fit_append(&x.view(), &y.view(), None, &mut rng)
                .unwrap();
        }
        let after = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        assert_eq!(
            before, after,
            "streaming one-row disk fit_append must not rewrite indexed_rows.bin"
        );
        assert_eq!(sur.model().expect("model").len(), 23);

        // wait_flush must not drain pending either.
        sur.wait_flush().unwrap();
        let after_wait = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        assert_eq!(before, after_wait, "wait_flush must not force ensure_sync");
    }

    #[test]
    fn bounded_observations() {
        let bounds = array![[0.0, 1.0]];
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 3,
            y_bounds: Some(bounds.clone()),
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(13);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let y = array![[0.1], [0.3], [0.4], [0.8], [0.5]];
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        assert_eq!(sur.y_bounds(), Some(&bounds));
        assert!((sur.y_row(0).unwrap()[0] - 0.1).abs() < 1e-12);
        let obs_y = sur.observations_y().unwrap().unwrap();
        for i in 0..5 {
            assert!((obs_y[[i, 0]] - y[[i, 0]]).abs() < 1e-12);
        }
        let pred = sur.predict(&array![[0.25, 0.75]].view()).unwrap();
        assert!(pred.mu[[0, 0]] > 0.0 && pred.mu[[0, 0]] < 1.0);
        let draws = sur
            .sample(&array![[0.25, 0.75]].view(), 4, &mut rng)
            .unwrap();
        assert!(draws.iter().all(|&v| v > 0.0 && v < 1.0));
    }

    #[test]
    fn auto_lengthscales() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            metric_learning: MetricLearning::Auto,
            unscaled_dims: vec![1],
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(23);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [0.5], [1.5]];
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        let ls = sur.lengthscales().unwrap();
        assert_eq!(ls.len(), 2);
        assert!((ls.mean().unwrap() - 1.0).abs() < 1e-12);
    }
}
