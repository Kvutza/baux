use std::borrow::Cow;
use std::path::PathBuf;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use super::{BayesOptimizer, ObservationDelta};
use crate::config::SurrogateConfig;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
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
    #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
    Disk {
        work_dir: PathBuf,
        store: Arc<DiskBpannEnnBackend>,
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
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        {
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
                    store: Arc::new(store),
                },
                yvar_mode: None,
            })
        }
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        {
            let _ = (work_dir, num_dim, num_metrics);
            Err(crate::backend::disk_unavailable())
        }
    }

    pub(crate) fn is_disk(&self) -> bool {
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        {
            matches!(self.backend, PosteriorBackend::Disk { .. })
        }
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        {
            false
        }
    }

    pub fn has_surrogate(&self) -> bool {
        matches!(self.backend, PosteriorBackend::Model(_))
    }

    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        match &self.backend {
            PosteriorBackend::Model(s) => Some(s.as_ref()),
            PosteriorBackend::Empirical { .. } => None,
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => None,
        }
    }

    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => Some(s.as_mut()),
            PosteriorBackend::Empirical { .. } => None,
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => None,
        }
    }

    pub fn check_variance(mut self, has_yvar: bool) -> Result<Self, ENNError> {
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
        Ok(self)
    }

    pub fn obs_count(&self) -> usize {
        match &self.backend {
            PosteriorBackend::Model(s) => s.observation_count().unwrap_or(0),
            PosteriorBackend::Empirical { x_rows, .. } => x_rows.len(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.len(),
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.row_x(idx),
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.row_y(idx),
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => Ok(Cow::Owned(store.slice_x(idx)?.to_vec())),
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => Ok(Cow::Owned(store.row_y(idx)?.to_vec())),
        }
    }

    pub fn unit_obs(&self) -> Option<Array2<f64>> {
        match &self.backend {
            PosteriorBackend::Model(s) => s.observations_x().ok().flatten(),
            PosteriorBackend::Empirical { x_rows, .. } => {
                (!x_rows.is_empty()).then(|| stack_rows(x_rows))
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => {
                let n = store.len();
                if n == 0 {
                    return None;
                }
                let rows: Vec<Array1<f64>> = (0..n).filter_map(|i| store.row_x(i).ok()).collect();
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => {
                let n = store.len();
                if n == 0 {
                    return None;
                }
                let rows: Vec<Array1<f64>> = (0..n).filter_map(|i| store.row_y(i).ok()).collect();
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

    pub fn append_empirical(&mut self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) {
        if let PosteriorBackend::Empirical { x_rows, y_rows } = &mut self.backend {
            for row in x.rows() {
                x_rows.push(row.to_owned());
            }
            for row in y.rows() {
                y_rows.push(row.to_owned());
            }
        }
    }

    pub fn append_disk(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        match &mut self.backend {
            PosteriorBackend::Disk { store, .. } => {
                let inner = (**store).clone();
                let updated = inner.append_rows(x, y, None)?;
                *store = Arc::new(updated);
            }
            _ => {}
        }
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        let _ = (x, y);
        Ok(())
    }

    pub fn append(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self = self.check_variance(yvar.is_some())?;
        match self.backend {
            PosteriorBackend::Empirical {
                mut x_rows,
                mut y_rows,
            } => {
                for row in x.rows() {
                    x_rows.push(row.to_owned());
                }
                for row in y.rows() {
                    y_rows.push(row.to_owned());
                }
                self.backend = PosteriorBackend::Empirical { x_rows, y_rows };
                Ok(self)
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { work_dir, store } => {
                let inner = Arc::try_unwrap(store).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.append_rows(x, y, yvar)?;
                self.backend = PosteriorBackend::Disk {
                    work_dir,
                    store: Arc::new(updated),
                };
                Ok(self)
            }
            PosteriorBackend::Model(surrogate) => {
                let updated = surrogate.with_append(x, y, yvar, rng)?;
                self.backend = PosteriorBackend::Model(updated);
                Ok(self)
            }
        }
    }

    pub fn fit_delta(
        mut self,
        delta: &ObservationDelta,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Option<f64>), ENNError> {
        let ncols = delta.y_new.ncols();
        if let Some(nm) = self.surrogate().and_then(|s| s.fitted_metrics()) {
            if nm != ncols {
                return Err(ENNError::InvalidParameter(format!(
                    "y has {ncols} metric columns but the fitted model has {nm}; changing output \
                     width is unsupported"
                )));
            }
        }
        if let PosteriorBackend::Model(surrogate) = self.backend {
            let start = std::time::Instant::now();
            let updated = surrogate.with_append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
            self.backend = PosteriorBackend::Model(updated);
            return Ok((self, Some(start.elapsed().as_secs_f64())));
        }
        Ok((self, None))
    }

    pub fn clear(mut self) -> Result<Self, ENNError> {
        match self.backend {
            PosteriorBackend::Model(s) => {
                let cleared = s.cleared()?;
                self.backend = PosteriorBackend::Model(cleared);
                Ok(self)
            }
            PosteriorBackend::Empirical {
                mut x_rows,
                mut y_rows,
            } => {
                x_rows.clear();
                y_rows.clear();
                self.backend = PosteriorBackend::Empirical { x_rows, y_rows };
                Ok(self)
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { work_dir, store } => {
                let (nd, nm) = (store.num_dim(), store.num_metrics());
                let _ = std::fs::remove_dir_all(&work_dir);
                let fresh = DiskBpannEnnBackend::new_thresholds(
                    work_dir.clone(),
                    nd,
                    nm,
                    usize::MAX,
                    usize::MAX,
                )?;
                self.backend = PosteriorBackend::Disk {
                    work_dir,
                    store: Arc::new(fresh),
                };
                Ok(self)
            }
        }
    }

    pub fn thaw(mut self, bytes: &[u8], pos: &mut usize) -> Result<Self, ENNError> {
        match self.backend {
            PosteriorBackend::Model(surrogate) => {
                let thawed = surrogate.with_thawed(bytes, pos)?;
                self.backend = PosteriorBackend::Model(thawed);
                Ok(self)
            }
            PosteriorBackend::Empirical { .. } => self.clear(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => self.clear(),
        }
    }

    pub fn schedule_flush(&mut self) -> Result<(), ENNError> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => {
                s.schedule_flush()?;
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => {
                let inner = (**store).clone();
                let updated = inner.schedule_flush()?;
                *store = Arc::new(updated);
            }
            PosteriorBackend::Empirical { .. } => {}
        }
        Ok(())
    }

    pub fn wait_flush(&mut self) -> Result<(), ENNError> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => {
                s.wait_flush()?;
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => {
                let inner = (**store).clone();
                let updated = inner.wait_flush()?;
                *store = Arc::new(updated);
            }
            PosteriorBackend::Empirical { .. } => {}
        }
        Ok(())
    }

    pub fn release_pages(&mut self) -> Result<(), ENNError> {
        match &mut self.backend {
            PosteriorBackend::Model(s) => {
                s.release_pages()?;
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => {
                let inner = (**store).clone();
                let updated = inner.release_pages()?;
                *store = Arc::new(updated);
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
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { work_dir, store } => PosteriorBackend::Disk {
                work_dir: work_dir.clone(),
                store: Arc::clone(store),
            },
            PosteriorBackend::Model(surrogate) => PosteriorBackend::Model(surrogate.clone_box()?),
        };
        Ok(Self {
            backend,
            yvar_mode: self.yvar_mode,
        })
    }

    pub fn fork(&self) -> Result<Self, ENNError> {
        self.try_clone()
    }

    pub(crate) fn hash_state(&self, buf: &mut Vec<u8>) {
        let hash_rows = |buf: &mut Vec<u8>| {
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
        };
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
                    hash_rows(buf);
                }
            }
            PosteriorBackend::Empirical { .. } => {
                buf.push(0);
                hash_rows(buf);
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => {
                buf.push(1);
                hash_rows(buf);
            }
        }
    }
}

impl Clone for PosteriorStore {
    fn clone(&self) -> Self {
        self.try_clone().expect("PosteriorStore clone failed")
    }
}

impl Default for PosteriorStore {
    fn default() -> Self {
        Self {
            backend: PosteriorBackend::Empirical {
                x_rows: Vec::new(),
                y_rows: Vec::new(),
            },
            yvar_mode: None,
        }
    }
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<PosteriorStore>();
    }
};

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
