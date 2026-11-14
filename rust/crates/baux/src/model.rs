//! Epistemic Nearest Neighbors (ENN) model implementation.
//!
//! Reference:
//! - Mehul Bafna, Siddhant Anand Jadhav, and David Sweet, "Taking the GP Out of
//!   the Loop", ICML 2026 (PMLR 306:5179-5199, arXiv:2506.12818).

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;

use crate::backend::{EnnBackend, EnnStorage};
use crate::error::ENNError;
use crate::index::{IndexDriver, disk_driver};

type InitStats = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
);

mod access;
mod inspection;
mod metric;
#[cfg(test)]
mod metric_tests;
pub use access::{EnnIndexAccess, EnnRowAccess};
pub use inspection::{ENN, ModelOptions};

impl ENN {
    fn validate_shapes(
        train_x: &Array2<f64>,
        train_y: &Array2<f64>,
        train_yvar: Option<&Array2<f64>>,
    ) -> Result<(), ENNError> {
        if train_x.nrows() != train_y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![train_y.nrows(), train_x.ncols()],
                got: vec![train_x.nrows(), train_x.ncols()],
            });
        }
        if let Some(yvar) = train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: train_y.shape().to_vec(),
                    got: yvar.shape().to_vec(),
                });
            }
        }
        Ok(())
    }

    fn init_stats(train_x: &Array2<f64>, train_y: &Array2<f64>, scale_x: bool) -> InitStats {
        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let (y_sum, y_sumsq) = column_sumsq(train_y.view());
        let y_scale = scale_moments(num_obs, num_metrics, &y_sum, &y_sumsq, 0.0);
        if scale_x {
            let (x_sum, x_sumsq) = column_sumsq(train_x.view());
            let x_scale = scale_moments(num_obs, num_dim, &x_sum, &x_sumsq, 1e-12);
            (y_scale, y_sum, y_sumsq, x_scale, x_sum, x_sumsq)
        } else {
            (
                y_scale,
                y_sum,
                y_sumsq,
                Array1::ones(num_dim),
                Array1::zeros(num_dim),
                Array1::zeros(num_dim),
            )
        }
    }

    /// Create a new ENN model (in-memory backend).
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Self::new_storage(
            train_x,
            train_y,
            train_yvar,
            scale_x,
            driver,
            EnnStorage::InMemory,
            None,
        )
    }

    /// Create a new ENN model with explicit storage backend.
    pub fn new_storage(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
    ) -> Result<Self, ENNError> {
        Self::new_options(
            train_x,
            train_y,
            train_yvar,
            ModelOptions {
                scale_x,
                driver,
                storage,
                work_dir,
                y_bounds: None,
            },
        )
    }

    pub fn new_options(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        options: ModelOptions,
    ) -> Result<Self, ENNError> {
        let ModelOptions {
            scale_x,
            driver,
            storage,
            work_dir,
            y_bounds,
        } = options;
        Self::validate_shapes(&train_x, &train_y, train_yvar.as_ref())?;
        let disk_work_dir = work_dir.clone().or_else(EnnStorage::work_env);
        let persisted_bounds = disk_work_dir.as_ref().and_then(|dir| {
            crate::y_bounds::load_metadata(dir, train_y.ncols())
                .ok()
                .flatten()
        });
        let y_bounds = match (y_bounds, persisted_bounds) {
            (Some(requested), Some(stored)) if requested != stored => {
                return Err(ENNError::InvalidParameter(
                    "requested y_bounds do not match persisted y_bounds".to_string(),
                ));
            }
            (Some(requested), _) => requested,
            (None, Some(stored)) => stored,
            (None, None) => crate::y_bounds::unbounded_bounds(train_y.ncols()),
        };
        crate::y_bounds::validate_bounds(&y_bounds, train_y.ncols())?;
        let bounded_outputs = !crate::y_bounds::identity_bounds(&y_bounds);
        let (train_y, train_yvar) = warp_training(train_y, train_yvar, &y_bounds, bounded_outputs)?;
        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let (y_scale, y_sum, y_sumsq, x_scale, x_sum, x_sumsq) =
            Self::init_stats(&train_x, &train_y, scale_x);
        let disk_reopen = matches!(storage, EnnStorage::Disk)
            && train_x.nrows() == 0
            && train_y.nrows() == 0
            && disk_work_dir
                .as_ref()
                .is_some_and(|p| p.join("metadata.json").exists());

        let backend = match storage {
            EnnStorage::InMemory => EnnBackend::new_memory(
                train_x,
                train_y,
                train_yvar,
                scale_x,
                x_scale.clone(),
                driver,
            )?,
            EnnStorage::Disk => {
                if !disk_driver(driver) {
                    return Err(ENNError::InvalidParameter(
                        "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
                    ));
                }
                let dir = work_dir.or_else(EnnStorage::work_env).ok_or_else(|| {
                    ENNError::InvalidParameter(
                        "Disk storage requires work_dir or BAUX_WORK_DIR".to_string(),
                    )
                })?;
                EnnBackend::new_disk(
                    dir,
                    train_x,
                    train_y,
                    train_yvar,
                    scale_x,
                    x_scale.clone(),
                    driver,
                )?
            }
        };

        let mut model = Self {
            metric: None,
            metric_fixed: false,
            built_scale: x_scale.clone(),
            unscaled_dims: Vec::new(),
            tied_groups: Vec::new(),
            backend,
            num_obs,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            y_scale,
            y_bounds,
            bounded_outputs,
            stiefel: None,
            affine: None,
            y_sum,
            y_sumsq,
            x_sum,
            x_sumsq,
            work_dir: disk_work_dir,
            dist_metric: baux::semantics::neighbor::DistanceMetric::Euclidean,
            scale_cache: std::sync::Arc::new(std::sync::OnceLock::new()),
        };
        if disk_reopen || model.num_obs != model.backend.len() {
            sync_backend(&mut model)?;
        }
        model.stiefel = model.stiefel.as_ref().map(|s| s.with_count(model.num_obs));
        model.y_metadata()?;
        Ok(model)
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
        soft_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        let stored_work_dir = work_dir.clone().or_else(EnnStorage::work_env);
        let backend = EnnBackend::new_empty(
            num_dim,
            num_metrics,
            driver,
            storage,
            work_dir,
            soft_threshold,
        )?;
        let y_bounds = stored_work_dir
            .as_ref()
            .map(|dir| crate::y_bounds::load_metadata(dir, num_metrics))
            .transpose()?
            .flatten()
            .unwrap_or_else(|| crate::y_bounds::unbounded_bounds(num_metrics));
        let mut model = Self {
            metric: None,
            metric_fixed: false,
            built_scale: Array1::ones(num_dim),
            unscaled_dims: Vec::new(),
            tied_groups: Vec::new(),
            backend,
            num_obs: 0,
            num_dim,
            num_metrics,
            scale_x: false,
            x_scale: Array1::ones(num_dim),
            y_scale: Array1::ones(num_metrics),
            bounded_outputs: !crate::y_bounds::identity_bounds(&y_bounds),
            y_bounds,
            stiefel: None,
            affine: None,
            y_sum: Array1::zeros(num_metrics),
            y_sumsq: Array1::zeros(num_metrics),
            x_sum: Array1::zeros(num_dim),
            x_sumsq: Array1::zeros(num_dim),
            work_dir: stored_work_dir,
            dist_metric: baux::semantics::neighbor::DistanceMetric::Euclidean,
            scale_cache: std::sync::Arc::new(std::sync::OnceLock::new()),
        };
        if model.backend.len() != 0 {
            sync_backend(&mut model)?;
        }
        Ok(model)
    }

    fn validate_add(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Option<ENNError> {
        if x.nrows() != y.nrows() || x.ncols() != self.num_dim || y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidShape {
                expected: vec![y.nrows(), self.num_metrics],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidParameter(format!(
                "y has {} metric columns but model expects {}",
                y.ncols(),
                self.num_metrics
            )));
        }
        match (yvar, self.rows().row_yvar(0).ok().flatten().is_some()) {
            (Some(yv), _) if yv.shape() != y.shape() => Some(ENNError::InvalidShape {
                expected: y.shape().to_vec(),
                got: yv.shape().to_vec(),
            }),
            (Some(_), false) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar provided but model has no existing yvar".to_string(),
            )),
            (None, true) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar must be provided if model has existing yvar".to_string(),
            )),
            _ => None,
        }
    }

    pub fn add(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if let Some(err) = self.validate_add(x, y, yvar) {
            return Err(err);
        }
        if self.metric.is_some() && x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "AUTO metric needs finite observation rows".into(),
            ));
        }
        if x.nrows() > 0 {
            self.backend.wait_flush()?;
            if self.bounded_outputs {
                let warped_y = crate::y_bounds::warp_y(*y, &self.y_bounds)?;
                let warped_yvar = yvar
                    .map(|yv| crate::y_bounds::warp_yvar(*y, *yv, &self.y_bounds))
                    .transpose()?;
                let warped_yvar_view = warped_yvar.as_ref().map(|v| v.view());
                self.backend
                    .append_rows(x, &warped_y.view(), warped_yvar_view.as_ref())?;
                accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, warped_y.view());
            } else {
                self.backend.append_rows(x, y, yvar)?;
                accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, y.view());
            }
            let n = self.backend.len();
            self.y_scale = scale_moments(n, self.num_metrics, &self.y_sum, &self.y_sumsq, 0.0);

            if self.scale_x && !self.metric_fixed {
                accumulate_columns(&mut self.x_sum, &mut self.x_sumsq, x.view());
                self.refresh_scale(n)?;
            }

            self.num_obs = n;
            self.scale_cache = std::sync::Arc::new(std::sync::OnceLock::new());
            self.stiefel = self.stiefel.as_ref().map(|s| s.with_count(self.num_obs));
            self.observe_metric(x, y)?;
        }
        Ok(())
    }

    pub fn append_cow(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        let mut next = self.clone();
        next.scale_cache = std::sync::Arc::new(std::sync::OnceLock::new());
        next.add(x, y, yvar)?;
        Ok(next)
    }

    /// Schedule a background index flush when pending rows exceed the disk
    /// threshold.
    pub fn schedule_flush(&self) -> Result<(), ENNError> {
        self.backend.schedule_flush()
    }

    /// Merge in-memory index fragments and persist a single on-disk BPANN
    /// index.
    pub fn persist_index(&self) -> Result<(), ENNError> {
        crate::backend::enn_index(&self.backend)?;
        self.y_metadata()
    }

    pub fn natural_rows(
        &self,
        indices: &[usize],
    ) -> Result<(Array2<f64>, Array2<f64>, Option<Array2<f64>>), ENNError> {
        let (x, y_z, yvar_z) = self.rows().train_rows(indices)?;
        if !self.bounded_outputs {
            return Ok((x, y_z, yvar_z));
        }
        let y = crate::y_bounds::inv_y(y_z.view(), &self.y_bounds);
        let yvar = yvar_z
            .map(|zv| crate::y_bounds::naturalize_yvar(y_z.view(), zv.view(), &self.y_bounds));
        Ok((x, y, yvar))
    }

    pub fn natural_y(&self, index: usize) -> Result<Array1<f64>, ENNError> {
        let y_z = self.rows().row_y(index)?;
        if !self.bounded_outputs {
            return Ok(y_z);
        }
        let y_z_row = y_z.insert_axis(ndarray::Axis(0));
        Ok(crate::y_bounds::inv_y(y_z_row.view(), &self.y_bounds)
            .row(0)
            .to_owned())
    }

    pub(crate) fn y_metadata(&self) -> Result<(), ENNError> {
        match self.work_dir.as_ref() {
            Some(dir) => crate::y_bounds::persist_metadata(dir, &self.y_bounds),
            None => Ok(()),
        }
    }

    pub fn scales_x(&self) -> bool {
        self.scale_x
    }

    pub fn neighbors(
        &self,
        x: &ArrayView2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> Result<Array2<usize>, ENNError> {
        if x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if k < 0 {
            return Err(ENNError::InvalidParameter(format!(
                "k must be non-negative, got {k}"
            )));
        }
        if self.num_obs == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }
        if exclude_nearest && self.num_obs <= 1 {
            return Err(ENNError::InvalidParameter(format!(
                "exclude_nearest=true requires at least 2 observations, got {}",
                self.num_obs
            )));
        }
        let search_k = if exclude_nearest {
            ((k + 1) as usize).min(self.num_obs)
        } else {
            (k as usize).min(self.num_obs)
        };
        if search_k == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }
        if !self.backend.defers_search() {
            self.ensure_sync()?;
        }
        let (_, idx_full) = self.backend.search(x, search_k as i32, exclude_nearest)?;
        let k_out = (k as usize).min(idx_full.ncols());
        let mut result = Array2::zeros((x.nrows(), k_out));
        for i in 0..x.nrows() {
            for j in 0..k_out {
                result[[i, j]] = idx_full[[i, j]] as usize;
            }
        }
        Ok(result)
    }

    pub(crate) fn backend_driver(&self) -> IndexDriver {
        self.backend.driver()
    }

    pub(crate) fn y_opt(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.backend.y_view()
    }

    pub(crate) fn x_opt(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.backend.x_view()
    }

    pub(crate) fn backend_search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        if !self.backend.defers_search() {
            self.ensure_sync()?;
        }
        self.backend.search(x, search_k, exclude_nearest)
    }

    pub(crate) fn freeze_model(&self, buf: &mut Vec<u8>) -> Result<(), ENNError> {
        use crate::snapshot_io::{write_arr, write_bool, write_len, write_mat};
        self.backend.wait_flush()?;
        let n = self.num_obs;
        let indices: Vec<usize> = (0..n).collect();
        let (x, y_stored, yvar_stored) = if n > 0 {
            self.rows().train_rows(&indices)?
        } else {
            (
                Array2::zeros((0, self.num_dim)),
                Array2::zeros((0, self.num_metrics)),
                None,
            )
        };
        write_len(buf, n);
        write_len(buf, self.num_dim);
        write_len(buf, self.num_metrics);
        write_mat(buf, &x);
        write_mat(buf, &y_stored);
        match yvar_stored {
            Some(ref yv) => {
                write_bool(buf, true);
                write_mat(buf, yv);
            }
            None => write_bool(buf, false),
        }
        write_bool(buf, self.scale_x);
        write_bool(buf, self.metric_fixed);
        write_arr(buf, &self.x_scale);
        write_arr(buf, &self.built_scale);
        write_arr(buf, &self.y_scale);
        write_arr(buf, &self.y_sum);
        write_arr(buf, &self.y_sumsq);
        write_arr(buf, &self.x_sum);
        write_arr(buf, &self.x_sumsq);
        match &self.metric {
            Some(m) => {
                write_bool(buf, true);
                m.freeze_metric(buf);
            }
            None => write_bool(buf, false),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn thaw_model(
        options: ModelOptions,
        tied: Vec<Vec<usize>>,
        unscaled: Vec<usize>,
        geometry: baux::semantics::neighbor::DistanceMetric,
        stiefel: Option<baux::semantics::stiefel::Stiefel>,
        affine: Option<baux::semantics::affine::Affine>,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, ENNError> {
        use crate::snapshot_io::{MAX_DIM, MAX_METRICS, MAX_OBS, bad_state};
        let n = reader.read_len(MAX_OBS)?;
        let num_dim = reader.read_len(MAX_DIM)?;
        let num_metrics = reader.read_len(MAX_METRICS)?;
        if num_dim == 0 || num_metrics == 0 {
            return Err(bad_state("frozen ENN model must have positive shape"));
        }
        crate::metric_weights::validate_tied(&tied, num_dim)?;
        if unscaled.iter().any(|&d| d >= num_dim) {
            return Err(bad_state("frozen ENN unscaled dimension out of range"));
        }
        let train_x = reader.read_mat(n, num_dim, true)?;
        let train_y = reader.read_mat(n, num_metrics, true)?;
        if train_x.shape() != [n, num_dim] || train_y.shape() != [n, num_metrics] {
            return Err(bad_state("frozen ENN matrix shape mismatch"));
        }
        let train_yvar = if reader.read_bool()? {
            let yv = reader.read_mat(n, num_metrics, true)?;
            if yv.shape() != [n, num_metrics] || yv.iter().any(|&v| v < 0.0) {
                return Err(bad_state("invalid frozen ENN yvar matrix"));
            }
            Some(yv)
        } else {
            None
        };
        let scale_x = reader.read_bool()?;
        let metric_fixed = reader.read_bool()?;
        let x_scale = reader.read_arr(num_dim, true)?;
        let built_scale = reader.read_arr(num_dim, true)?;
        let y_scale = reader.read_arr(num_metrics, true)?;
        if x_scale.iter().chain(built_scale.iter()).any(|&v| v <= 0.0)
            || y_scale.iter().any(|&v| v <= 0.0)
        {
            return Err(bad_state("frozen ENN scales must be positive"));
        }
        let y_sum = reader.read_arr(num_metrics, true)?;
        let y_sumsq = reader.read_arr(num_metrics, true)?;
        let x_sum = reader.read_arr(num_dim, true)?;
        let x_sumsq = reader.read_arr(num_dim, true)?;
        let metric = if reader.read_bool()? {
            let mut auto_unscaled: Vec<usize> = tied.iter().flatten().copied().collect();
            auto_unscaled.extend(unscaled.iter().copied());
            auto_unscaled.sort_unstable();
            auto_unscaled.dedup();
            let only_extra: Vec<usize> = unscaled
                .iter()
                .copied()
                .filter(|j| !tied.iter().flatten().any(|t| t == j))
                .collect();
            Some(crate::metric_auto::AutoMetric::thaw_metric(
                num_dim,
                num_metrics,
                tied.clone(),
                only_extra,
                reader,
            )?)
        } else {
            None
        };

        let ModelOptions {
            driver,
            storage,
            work_dir,
            y_bounds,
            ..
        } = options;
        let disk_work_dir = work_dir.clone().or_else(EnnStorage::work_env);
        let y_bounds = match y_bounds {
            Some(requested) => requested,
            None => crate::y_bounds::unbounded_bounds(num_metrics),
        };
        crate::y_bounds::validate_bounds(&y_bounds, num_metrics)?;
        let bounded_outputs = !crate::y_bounds::identity_bounds(&y_bounds);
        let backend = match storage {
            EnnStorage::InMemory => EnnBackend::new_memory(
                train_x,
                train_y,
                train_yvar,
                scale_x,
                x_scale.clone(),
                driver,
            )?,
            EnnStorage::Disk => {
                if !disk_driver(driver) {
                    return Err(bad_state("Disk storage requires IndexDriver::BpAnnDisk"));
                }
                let dir = work_dir
                    .or_else(EnnStorage::work_env)
                    .ok_or_else(|| bad_state("Disk storage requires work_dir or BAUX_WORK_DIR"))?;
                EnnBackend::new_disk(
                    dir,
                    train_x,
                    train_y,
                    train_yvar,
                    scale_x,
                    x_scale.clone(),
                    driver,
                )?
            }
        };
        let mut model_unscaled = unscaled;
        if metric.is_some() {
            for &j in tied.iter().flatten() {
                if !model_unscaled.contains(&j) {
                    model_unscaled.push(j);
                }
            }
            model_unscaled.sort_unstable();
            model_unscaled.dedup();
        }
        let mut model = Self {
            metric,
            metric_fixed,
            built_scale,
            unscaled_dims: model_unscaled,
            tied_groups: tied,
            backend,
            num_obs: n,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            y_scale,
            y_bounds,
            bounded_outputs,
            stiefel: None,
            affine: None,
            y_sum,
            y_sumsq,
            x_sum,
            x_sumsq,
            work_dir: disk_work_dir,
            dist_metric: geometry,
            scale_cache: std::sync::Arc::new(std::sync::OnceLock::new()),
        };
        if let Some(st) = stiefel {
            model = model.with_stiefel(st)?;
        }
        if let Some(aff) = affine {
            model = model.with_affine(aff)?;
        }
        model.y_metadata()?;
        Ok(model)
    }

    pub(crate) fn hash_model(&self, buf: &mut Vec<u8>) {
        let _ = self.freeze_model(buf);
    }
}

fn warp_training(
    train_y: Array2<f64>,
    train_yvar: Option<Array2<f64>>,
    y_bounds: &Array2<f64>,
    bounded_outputs: bool,
) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
    let result = if !bounded_outputs {
        (train_y, train_yvar)
    } else {
        let warped_y = crate::y_bounds::warp_y(train_y.view(), y_bounds)?;
        let warped_yvar = match train_yvar {
            Some(yv) => Some(crate::y_bounds::warp_yvar(
                train_y.view(),
                yv.view(),
                y_bounds,
            )?),
            None => None,
        };
        (warped_y, warped_yvar)
    };
    Ok(result)
}

const STATS_CHUNK: usize = 1 << 16;

/// Rebuild observation count and scale moments from persisted backend rows
/// (disk reopen).
fn sync_backend(model: &mut ENN) -> Result<(), ENNError> {
    model.num_dim = model.backend.num_dim();
    model.num_metrics = model.backend.num_metrics();
    let n = model.backend.len();
    model.num_obs = n;
    if n == 0 {
        return Ok(());
    }
    let mut y_sum = Array1::zeros(model.num_metrics);
    let mut y_sumsq = Array1::zeros(model.num_metrics);
    let mut x_sum = Array1::zeros(model.num_dim);
    let mut x_sumsq = Array1::zeros(model.num_dim);
    let mut start = 0usize;
    while start < n {
        let end = (start + STATS_CHUNK).min(n);
        let indices: Vec<usize> = (start..end).collect();
        let (x, y, _) = model.backend.train_rows(&indices)?;
        accumulate_columns(&mut y_sum, &mut y_sumsq, y.view());
        if model.scale_x {
            accumulate_columns(&mut x_sum, &mut x_sumsq, x.view());
        }
        start = end;
    }
    model.y_sum = y_sum;
    model.y_sumsq = y_sumsq;
    model.y_scale = scale_moments(n, model.num_metrics, &model.y_sum, &model.y_sumsq, 0.0);
    if model.scale_x {
        model.x_sum = x_sum;
        model.x_sumsq = x_sumsq;
        model.x_scale = model.data_scale(n);
        model.built_scale = model.x_scale.clone();
    }
    model.backend.ensure_sync(model.scale_x, &model.x_scale)?;
    Ok(())
}

fn column_sumsq(a: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let ncol = a.ncols();
    let mut sum = Array1::zeros(ncol);
    let mut sumsq = Array1::zeros(ncol);
    for row in a.axis_iter(ndarray::Axis(0)) {
        for j in 0..ncol {
            let v = row[j];
            sum[j] += v;
            sumsq[j] += v * v;
        }
    }
    (sum, sumsq)
}

fn accumulate_columns(sum: &mut Array1<f64>, sumsq: &mut Array1<f64>, extra: ArrayView2<f64>) {
    let ncol = extra.ncols();
    for row in extra.axis_iter(ndarray::Axis(0)) {
        for j in 0..ncol {
            let v = row[j];
            sum[j] += v;
            sumsq[j] += v * v;
        }
    }
}

pub(crate) fn scale_moments(
    n: usize,
    ncol: usize,
    sum: &Array1<f64>,
    sumsq: &Array1<f64>,
    min_std: f64,
) -> Array1<f64> {
    if n < 2 {
        return Array1::ones(ncol);
    }
    let nf = n as f64;
    Array1::from_iter((0..ncol).map(|j| {
        let mean = sum[j] / nf;
        let var = (sumsq[j] / nf - mean * mean).max(0.0);
        let std = var.sqrt();
        if std.is_finite() && std > min_std {
            std
        } else {
            1.0
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::row_storage::RowStorage;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn test_enncreation() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        assert_eq!(model.len(), 4);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_ennadd() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let mut model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        model
            .add(&array![[0.0, 1.0]].view(), &array![[1.0]].view(), None)
            .unwrap();
        assert_eq!(model.len(), 3);
    }

    #[test]
    fn storage_helpers() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let mut storage = RowStorage::from_array(rows.clone());
        assert_eq!(storage.nrows(), 2);
        storage.push_rows(&array![[5.0, 6.0]].view()).unwrap();
        assert_eq!(storage.nrows(), 3);
        let (sum, sumsq) = column_sumsq(rows.view());
        let mut sum2 = sum.clone();
        let mut sumsq2 = sumsq.clone();
        accumulate_columns(&mut sum2, &mut sumsq2, array![[0.0, 0.0]].view());
        let scale = scale_moments(2, 2, &sum, &sumsq, 1e-9);
        assert_eq!(scale.len(), 2);
    }

    #[test]
    fn test_004() {
        let mut model =
            ENN::new_empty(2, 1, IndexDriver::Exact, EnnStorage::InMemory, None, None).unwrap();
        model
            .add(&array![[1.0, 2.0]].view(), &array![[3.0]].view(), None)
            .unwrap();
        let x = model.rows().row_x(0).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        let y = model.rows().row_y(0).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn internal_index() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        assert!(model.x_opt().is_some());
        assert!(model.y_opt().is_some());
        assert_eq!(model.index_access().len(), 2);
    }

    #[test]
    fn accessors_report() {
        let mut model = ENN::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        assert!(!model.scales_x());
        assert_eq!(model.backend_driver(), IndexDriver::Exact);
        let _ = model.x_scale();
        model
            .add(&array![[0.5, 0.5]].view(), &array![[0.5]].view(), None)
            .unwrap();
    }

    #[test]
    fn bounded_trip() {
        let bounds = array![[0.0, 1.0]];
        let mut model = ENN::new_options(
            array![[0.0], [1.0]],
            array![[0.2], [0.8]],
            None,
            ModelOptions {
                y_bounds: Some(bounds.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(model.bounded_outputs());
        assert_eq!(model.y_bounds(), &bounds);
        assert!((model.rows().row_y(0).unwrap()[0] - 0.2).abs() > 1e-6);
        assert!((model.natural_y(0).unwrap()[0] - 0.2).abs() < 1e-12);

        model
            .add(&array![[2.0]].view(), &array![[0.6]].view(), None)
            .unwrap();
        assert!((model.natural_y(2).unwrap()[0] - 0.6).abs() < 1e-12);
        assert!(
            model
                .add(&array![[3.0]].view(), &array![[1.0]].view(), None)
                .is_err()
        );
    }

    #[test]
    fn disk_reopen() {
        let dir = TempDir::new().unwrap();
        let bounds = array![[0.0, 1.0]];
        let model = ENN::new_options(
            array![[0.0], [1.0]],
            array![[0.2], [0.8]],
            None,
            ModelOptions {
                driver: IndexDriver::BpAnnDisk,
                storage: EnnStorage::Disk,
                work_dir: Some(dir.path().to_path_buf()),
                y_bounds: Some(bounds.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        model.persist_index().unwrap();

        let reopened = ENN::new_options(
            Array2::zeros((0, 1)),
            Array2::zeros((0, 1)),
            None,
            ModelOptions {
                driver: IndexDriver::BpAnnDisk,
                storage: EnnStorage::Disk,
                work_dir: Some(dir.path().to_path_buf()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(reopened.y_bounds(), &bounds);
        assert!((reopened.natural_y(0).unwrap()[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cow_append() {
        use crate::params::{ENNParams, PosteriorFlags};
        use crate::traits::PosteriorComputation;

        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let m1 = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let st = baux::semantics::stiefel::Stiefel::new(vec![1.0, 0.0], 2, 1).unwrap();
        let aff =
            baux::semantics::affine::Affine::new(vec![0.5, -0.5], vec![2.0, 1.0], vec![1.5, 0.5])
                .unwrap();
        let m1 = m1.with_stiefel(st).unwrap().with_affine(aff).unwrap();
        assert_eq!(m1.stiefel().unwrap().count(), 3);

        let m2 = m1
            .append_cow(&array![[1.0, 1.0]].view(), &array![[4.0, 8.0]].view(), None)
            .unwrap();
        assert_eq!(m1.len(), 3);
        assert_eq!(m2.len(), 4);
        assert_eq!(m1.stiefel().unwrap().count(), 3);
        assert_eq!(m2.stiefel().unwrap().count(), 4);

        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let q = array![[1.0, 1.0]];
        let p1 = m1.posterior(&q.view(), &params, &flags).unwrap();
        let p2 = m2.posterior(&q.view(), &params, &flags).unwrap();
        assert_eq!(p1.mu.shape(), &[1, 2]);
        assert_eq!(p2.mu.shape(), &[1, 2]);
        assert!((p1.mu[[0, 0]] - p2.mu[[0, 0]]).abs() > 1e-6);
    }
}
