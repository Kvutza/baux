use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use super::{BayesOptimizer, ObservationDelta};
use crate::config::SurrogateConfig;
use crate::disk_bpann::DiskBpannEnnBackend;
use crate::error::ENNError;
use crate::optimizer_dispatch::{VarianceGate, variance_gate};
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};

/// Closed storage backend: empirical observation vectors (RAM or disk mmap) or
/// an indexed surrogate model.
pub enum PosteriorBackend {
    Empirical {
        x_rows: Vec<Array1<f64>>,
        y_rows: Vec<Array1<f64>>,
    },
    Disk {
        work_dir: PathBuf,
        store: Mutex<DiskBpannEnnBackend>,
    },
    Model(BoxedSurrogate),
}

/// Unified posterior and observation store for the ask/tell optimizer.
pub struct PosteriorStore {
    pub(crate) backend: PosteriorBackend,
    pub(crate) yvar_mode: Option<bool>,
}

impl PosteriorStore {
    pub fn new(config: &SurrogateConfig, provided: Option<BoxedSurrogate>) -> Self {
        let model = provided.or_else(|| match config {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        });
        let backend = match model {
            Some(surrogate) => PosteriorBackend::Model(surrogate),
            None => PosteriorBackend::Empirical {
                x_rows: Vec::new(),
                y_rows: Vec::new(),
            },
        };
        Self {
            backend,
            yvar_mode: None,
        }
    }

    pub fn new_disk(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        let store = DiskBpannEnnBackend::new_thresholds(
            work_dir.clone(),
            num_dim,
            num_metrics,
            usize::MAX,
            usize::MAX,
        )?;
        Ok(Self {
            backend: PosteriorBackend::Disk {
                work_dir,
                store: Mutex::new(store),
            },
            yvar_mode: None,
        })
    }

    pub fn has_surrogate(&self) -> bool {
        matches!(self.backend, PosteriorBackend::Model(_))
    }

    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        match &self.backend {
            PosteriorBackend::Model(s) => Some(s.as_ref()),
            PosteriorBackend::Empirical { .. } | PosteriorBackend::Disk { .. } => None,
        }
    }

    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => Some(s.as_mut()),
            PosteriorBackend::Empirical { .. } | PosteriorBackend::Disk { .. } => None,
        }
    }

    pub fn check_variance(&mut self, has_yvar: bool) -> Result<(), ENNError> {
        match variance_gate(self.yvar_mode, has_yvar) {
            VarianceGate::Set(value) => self.yvar_mode = Some(value),
            VarianceGate::Reject => {
                return Err(ENNError::InvalidParameter(format!(
                    "y_var must be {} on every tell()",
                    if self.yvar_mode == Some(true) {
                        "provided"
                    } else {
                        "omitted"
                    }
                )));
            }
            VarianceGate::Keep => {}
        }
        Ok(())
    }

    pub fn obs_count(&self) -> usize {
        match &self.backend {
            PosteriorBackend::Model(s) => s.observation_count().unwrap_or(0),
            PosteriorBackend::Empirical { x_rows, .. } => x_rows.len(),
            PosteriorBackend::Disk { store, .. } => store.lock().map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.x_row(idx),
            PosteriorBackend::Empirical { x_rows, .. } => {
                x_rows.get(idx).cloned().ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })
            }
            PosteriorBackend::Disk { store, .. } => store
                .lock()
                .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?
                .row_x(idx),
        }
    }

    pub fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.y_row(idx),
            PosteriorBackend::Empirical { y_rows, .. } => {
                y_rows.get(idx).cloned().ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })
            }
            PosteriorBackend::Disk { store, .. } => store
                .lock()
                .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?
                .row_y(idx),
        }
    }

    /// Zero-copy borrowed row slice when empirical, or surrogate-provided
    /// slice.
    pub fn x_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.x_slice(idx),
            PosteriorBackend::Empirical { x_rows, .. } => {
                let row = x_rows.get(idx).ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })?;
                Ok(match row.as_slice() {
                    Some(slice) => Cow::Borrowed(slice),
                    None => Cow::Owned(row.to_vec()),
                })
            }
            PosteriorBackend::Disk { store, .. } => {
                let g = store
                    .lock()
                    .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?;
                Ok(Cow::Owned(g.slice_x(idx)?.to_vec()))
            }
        }
    }

    /// Zero-copy borrowed target slice when empirical, or surrogate-provided
    /// slice.
    pub fn y_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.y_slice(idx),
            PosteriorBackend::Empirical { y_rows, .. } => {
                let row = y_rows.get(idx).ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })?;
                Ok(match row.as_slice() {
                    Some(slice) => Cow::Borrowed(slice),
                    None => Cow::Owned(row.to_vec()),
                })
            }
            PosteriorBackend::Disk { store, .. } => Ok(Cow::Owned(
                store
                    .lock()
                    .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?
                    .row_y(idx)?
                    .to_vec(),
            )),
        }
    }

    pub fn unit_obs(&self) -> Option<Array2<f64>> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.observations_x().ok().flatten(),
            PosteriorBackend::Empirical { x_rows, .. } => {
                (!x_rows.is_empty()).then(|| stack_rows(x_rows))
            }
            PosteriorBackend::Disk { store, .. } => {
                let g = store.lock().ok()?;
                let n = g.len();
                if n == 0 {
                    return None;
                }
                let rows: Vec<Array1<f64>> = (0..n).filter_map(|i| g.row_x(i).ok()).collect();
                (rows.len() == n).then(|| stack_rows(&rows))
            }
        }
    }

    pub fn y_obs(&self) -> Option<Array2<f64>> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.observations_y().ok().flatten(),
            PosteriorBackend::Empirical { y_rows, .. } => {
                (!y_rows.is_empty()).then(|| stack_rows(y_rows))
            }
            PosteriorBackend::Disk { store, .. } => {
                let g = store.lock().ok()?;
                let n = g.len();
                if n == 0 {
                    return None;
                }
                let rows: Vec<Array1<f64>> = (0..n).filter_map(|i| g.row_y(i).ok()).collect();
                (rows.len() == n).then(|| stack_rows(&rows))
            }
        }
    }

    pub fn push_empirical(&mut self, x_row: Array1<f64>, y_row: Array1<f64>) {
        if let PosteriorBackend::Empirical { x_rows, y_rows } = &mut self.backend {
            x_rows.push(x_row);
            y_rows.push(y_row);
        }
    }

    pub fn append_disk(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if let PosteriorBackend::Disk { store, .. } = &mut self.backend {
            let g = store
                .get_mut()
                .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?;
            g.append_rows(x, y, None)?;
        }
        Ok(())
    }

    pub fn fit_delta(
        &mut self,
        delta: &ObservationDelta,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Option<f64>, ENNError> {
        let ncols = delta.y_new.ncols();
        if let Some(nm) = self.surrogate().and_then(|s| s.fitted_metrics()) {
            if nm != ncols {
                return Err(ENNError::InvalidParameter(format!(
                    "y has {ncols} metric columns but the fitted model has {nm}; changing output \
                     width is unsupported"
                )));
            }
        }
        if let PosteriorBackend::Model(surrogate) = &mut self.backend {
            let start = std::time::Instant::now();
            surrogate.fit_append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
            return Ok(Some(start.elapsed().as_secs_f64()));
        }
        Ok(None)
    }

    pub fn clear(&mut self) -> Result<(), ENNError> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => s.clear_observations(),
            PosteriorBackend::Empirical { x_rows, y_rows } => {
                x_rows.clear();
                y_rows.clear();
                Ok(())
            }
            PosteriorBackend::Disk { work_dir, store } => {
                let (nd, nm) = {
                    let g = store.get_mut().map_err(|_| {
                        ENNError::InvalidParameter("disk store lock poisoned".into())
                    })?;
                    (g.num_dim(), g.num_metrics())
                };
                let _ = std::fs::remove_dir_all(&*work_dir);
                let fresh = DiskBpannEnnBackend::new_thresholds(
                    work_dir.clone(),
                    nd,
                    nm,
                    usize::MAX,
                    usize::MAX,
                )?;
                *store = Mutex::new(fresh);
                Ok(())
            }
        }
    }

    pub fn schedule_flush(&self) -> Result<(), ENNError> {
        if let PosteriorBackend::Model(s) = &self.backend {
            s.schedule_flush()?;
        }
        Ok(())
    }

    pub fn wait_flush(&self) -> Result<(), ENNError> {
        if let PosteriorBackend::Model(s) = &self.backend {
            s.wait_flush()?;
        }
        Ok(())
    }

    pub fn release_pages(&self) -> Result<(), ENNError> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.release_pages()?,
            PosteriorBackend::Disk { store, .. } => {
                store
                    .lock()
                    .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?
                    .release_pages()?;
            }
            PosteriorBackend::Empirical { .. } => {}
        }
        Ok(())
    }

    pub fn try_clone(&self) -> Result<Self, ENNError> {
        let backend = match &self.backend {
            PosteriorBackend::Empirical { x_rows, y_rows } => PosteriorBackend::Empirical {
                x_rows: x_rows.clone(),
                y_rows: y_rows.clone(),
            },
            PosteriorBackend::Disk { work_dir, store } => {
                let g = store
                    .lock()
                    .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".into()))?;
                let clone_dir = work_dir.with_extension(format!(
                    "fork-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                let mut cloned = DiskBpannEnnBackend::new_thresholds(
                    clone_dir.clone(),
                    g.num_dim(),
                    g.num_metrics(),
                    usize::MAX,
                    usize::MAX,
                )?;
                for i in 0..g.len() {
                    let xr = g.row_x(i)?.insert_axis(ndarray::Axis(0));
                    let yr = g.row_y(i)?.insert_axis(ndarray::Axis(0));
                    cloned.append_rows(&xr.view(), &yr.view(), None)?;
                }
                PosteriorBackend::Disk {
                    work_dir: clone_dir,
                    store: Mutex::new(cloned),
                }
            }
            PosteriorBackend::Model(surrogate) => PosteriorBackend::Model(surrogate.clone_box()?),
        };
        Ok(Self {
            backend,
            yvar_mode: self.yvar_mode,
        })
    }

    pub(crate) fn hash_state(&self, buf: &mut Vec<u8>) {
        buf.push(match self.yvar_mode {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
        match &self.backend {
            PosteriorBackend::Model(s) => {
                buf.push(2);
                let before = buf.len();
                s.hash_surrogate(buf);
                if buf.len() == before {
                    let n = self.obs_count();
                    buf.extend_from_slice(&n.to_le_bytes());
                    for i in 0..n {
                        if let Ok(x_row) = self.x_slice(i) {
                            for &v in x_row.iter() {
                                buf.extend_from_slice(&v.to_bits().to_le_bytes());
                            }
                        }
                        if let Ok(y_row) = self.y_slice(i) {
                            for &v in y_row.iter() {
                                buf.extend_from_slice(&v.to_bits().to_le_bytes());
                            }
                        }
                    }
                }
            }
            PosteriorBackend::Empirical { .. } | PosteriorBackend::Disk { .. } => {
                buf.push(if matches!(self.backend, PosteriorBackend::Disk { .. }) {
                    1
                } else {
                    0
                });
                let n = self.obs_count();
                buf.extend_from_slice(&n.to_le_bytes());
                for i in 0..n {
                    if let Ok(x_row) = self.x_slice(i) {
                        for &v in x_row.iter() {
                            buf.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                    }
                    if let Ok(y_row) = self.y_slice(i) {
                        for &v in y_row.iter() {
                            buf.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
    }
}

/// Public accessor for optimizer observation row reads.
pub struct ObsAccess<'a> {
    opt: &'a BayesOptimizer,
}

impl<'a> ObsAccess<'a> {
    pub fn observations_empty(&self) -> bool {
        self.opt.posterior.obs_count() == 0
    }

    pub fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        self.opt.posterior.x_row(idx)
    }

    pub fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        self.opt.posterior.y_row(idx)
    }
}

impl BayesOptimizer {
    /// Row-level observation access (unified posterior store).
    pub fn obs_access(&self) -> ObsAccess<'_> {
        ObsAccess { opt: self }
    }
}

pub fn stack_rows(vecs: &[Array1<f64>]) -> Array2<f64> {
    let n = vecs.len();
    let d = vecs[0].len();
    let mut result = Array2::zeros((n, d));
    for (i, v) in vecs.iter().enumerate() {
        for j in 0..d {
            result[[i, j]] = v[j];
        }
    }
    result
}
