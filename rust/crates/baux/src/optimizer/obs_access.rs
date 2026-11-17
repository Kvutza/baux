use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use super::{BayesOptimizer, ObservationDelta};
use crate::config::SurrogateConfig;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
use crate::disk_bpann::DiskBpannEnnBackend;
use crate::error::ENNError;
use crate::optimizer_dispatch::{VarianceGate, variance_gate};
use crate::scratch::ImLog;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};

/// Closed storage backend: persistent [`ImLog`] observation chunks, disk mmap,
/// `Arc`-shared built-in [`ENNSurrogate`], or `Arc`-shared external surrogate
/// (`PythonSurrogateAdapter` / `baux.gp()`).
#[derive(Clone)]
pub enum PosteriorBackend {
    Empirical {
        x_log: ImLog,
        y_log: ImLog,
    },
    #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
    Disk {
        work_dir: PathBuf,
        store: Arc<DiskBpannEnnBackend>,
    },
    Enn(Arc<ENNSurrogate>),
    External(Arc<BoxedSurrogate>),
}

/// Unified posterior and observation store for the ask/tell optimizer.
#[derive(Clone)]
pub struct PosteriorStore {
    pub(crate) backend: PosteriorBackend,
    pub(crate) yvar_mode: Option<bool>,
}

impl PosteriorStore {
    pub fn new(config: &SurrogateConfig, provided: Option<BoxedSurrogate>) -> Self {
        let backend = match provided {
            Some(surrogate) => PosteriorBackend::External(Arc::new(surrogate)),
            None => match config {
                SurrogateConfig::ENN(enn_config) => {
                    PosteriorBackend::Enn(Arc::new(ENNSurrogate::new(enn_config.clone())))
                }
                SurrogateConfig::None => PosteriorBackend::Empirical {
                    x_log: ImLog::empty(),
                    y_log: ImLog::empty(),
                },
            },
        };
        Self {
            backend,
            yvar_mode: None,
        }
    }

    pub fn new_enn(enn: ENNSurrogate) -> Self {
        Self {
            backend: PosteriorBackend::Enn(Arc::new(enn)),
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

    pub fn shares_storage(&self, other: &Self) -> bool {
        match (&self.backend, &other.backend) {
            (
                PosteriorBackend::Empirical {
                    x_log: x1,
                    y_log: y1,
                },
                PosteriorBackend::Empirical {
                    x_log: x2,
                    y_log: y2,
                },
            ) => x1.shares_storage(x2) && y1.shares_storage(y2),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            (
                PosteriorBackend::Disk { store: s1, .. },
                PosteriorBackend::Disk { store: s2, .. },
            ) => Arc::ptr_eq(s1, s2),
            (PosteriorBackend::Enn(e1), PosteriorBackend::Enn(e2)) => Arc::ptr_eq(e1, e2),
            (PosteriorBackend::External(x1), PosteriorBackend::External(x2)) => Arc::ptr_eq(x1, x2),
            _ => false,
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
        matches!(
            self.backend,
            PosteriorBackend::Enn(_) | PosteriorBackend::External(_)
        )
    }

    pub fn enn(&self) -> Option<&ENNSurrogate> {
        match &self.backend {
            PosteriorBackend::Enn(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        match &self.backend {
            PosteriorBackend::Enn(s) => Some(s.as_ref()),
            PosteriorBackend::External(s) => Some(s.as_ref().as_ref()),
            PosteriorBackend::Empirical { .. } => None,
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => None,
        }
    }

    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match &mut self.backend {
            PosteriorBackend::Enn(s) => {
                let inner: &mut ENNSurrogate = Arc::make_mut(s);
                Some(inner)
            }
            PosteriorBackend::External(s) => {
                if Arc::strong_count(s) > 1 {
                    let cloned = s.clone_box().ok()?;
                    *s = Arc::new(cloned);
                }
                match Arc::get_mut(s) {
                    Some(boxed) => Some(boxed.as_mut()),
                    None => None,
                }
            }
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
            PosteriorBackend::Enn(s) => s.observation_count().unwrap_or(0),
            PosteriorBackend::External(s) => s.observation_count().unwrap_or(0),
            PosteriorBackend::Empirical { x_log, .. } => x_log.nrows(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.len(),
        }
    }

    pub fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.backend {
            PosteriorBackend::Enn(s) => s.x_row(idx),
            PosteriorBackend::External(s) => s.x_row(idx),
            PosteriorBackend::Empirical { x_log, .. } => x_log.checked_row(idx),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.row_x(idx),
        }
    }

    pub fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.backend {
            PosteriorBackend::Enn(s) => s.y_row(idx),
            PosteriorBackend::External(s) => s.y_row(idx),
            PosteriorBackend::Empirical { y_log, .. } => y_log.checked_row(idx),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => store.row_y(idx),
        }
    }

    /// Zero-copy borrowed row slice when empirical or in-memory ENN/adapter, or
    /// owned fallback on disk.
    pub fn x_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        match &self.backend {
            PosteriorBackend::Enn(s) => s.x_slice(idx),
            PosteriorBackend::External(s) => s.x_slice(idx),
            PosteriorBackend::Empirical { x_log, .. } => {
                x_log.checked_slice(idx).map(Cow::Borrowed)
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => Ok(Cow::Owned(store.slice_x(idx)?.to_vec())),
        }
    }

    /// Zero-copy borrowed target slice when empirical or in-memory ENN/adapter,
    /// or owned fallback on disk/warped outputs.
    pub fn y_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        match &self.backend {
            PosteriorBackend::Enn(s) => s.y_slice(idx),
            PosteriorBackend::External(s) => s.y_slice(idx),
            PosteriorBackend::Empirical { y_log, .. } => {
                y_log.checked_slice(idx).map(Cow::Borrowed)
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { store, .. } => Ok(Cow::Owned(store.row_y(idx)?.to_vec())),
        }
    }

    pub fn unit_obs(&self) -> Option<Array2<f64>> {
        match &self.backend {
            PosteriorBackend::Enn(s) => s.observations_x().ok().flatten(),
            PosteriorBackend::External(s) => s.observations_x().ok().flatten(),
            PosteriorBackend::Empirical { x_log, .. } => {
                (!x_log.is_empty()).then(|| x_log.to_array())
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
            PosteriorBackend::Enn(s) => s.observations_y().ok().flatten(),
            PosteriorBackend::External(s) => s.observations_y().ok().flatten(),
            PosteriorBackend::Empirical { y_log, .. } => {
                (!y_log.is_empty()).then(|| y_log.to_array())
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
        if let PosteriorBackend::Empirical { x_log, y_log } = &mut self.backend {
            let prev_x = std::mem::take(x_log);
            let prev_y = std::mem::take(y_log);
            *x_log = prev_x
                .push_slice(x_row.as_slice().unwrap_or(&x_row.to_vec()))
                .expect("empirical x_row shape mismatch");
            *y_log = prev_y
                .push_slice(y_row.as_slice().unwrap_or(&y_row.to_vec()))
                .expect("empirical y_row shape mismatch");
        }
    }

    pub fn append_empirical(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if let PosteriorBackend::Empirical { x_log, y_log } = &mut self.backend {
            let prev_x = std::mem::take(x_log);
            let prev_y = std::mem::take(y_log);
            *x_log = prev_x.push_rows(x)?;
            *y_log = prev_y.push_rows(y)?;
        }
        Ok(())
    }

    pub fn append_disk(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        if let PosteriorBackend::Disk { store, .. } = &mut self.backend {
            let taken = std::mem::replace(store, Arc::clone(store));
            let inner = Arc::try_unwrap(taken).unwrap_or_else(|arc| (*arc).clone());
            let updated = inner.append_rows(x, y, None)?;
            *store = Arc::new(updated);
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
            PosteriorBackend::Empirical { x_log, y_log } => {
                self.backend = PosteriorBackend::Empirical {
                    x_log: x_log.push_rows(x)?,
                    y_log: y_log.push_rows(y)?,
                };
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
            PosteriorBackend::Enn(enn) => {
                let inner = Arc::try_unwrap(enn).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.append(x, y, yvar, rng)?;
                self.backend = PosteriorBackend::Enn(Arc::new(updated));
                Ok(self)
            }
            PosteriorBackend::External(ext) => {
                let boxed = match Arc::try_unwrap(ext) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                let updated = boxed.with_append(x, y, yvar, rng)?;
                self.backend = PosteriorBackend::External(Arc::new(updated));
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
        match self.backend {
            PosteriorBackend::Enn(enn) => {
                let start = std::time::Instant::now();
                let inner = Arc::try_unwrap(enn).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
                self.backend = PosteriorBackend::Enn(Arc::new(updated));
                Ok((self, Some(start.elapsed().as_secs_f64())))
            }
            PosteriorBackend::External(ext) => {
                let start = std::time::Instant::now();
                let boxed = match Arc::try_unwrap(ext) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                let updated = boxed.with_append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
                self.backend = PosteriorBackend::External(Arc::new(updated));
                Ok((self, Some(start.elapsed().as_secs_f64())))
            }
            _ => Ok((self, None)),
        }
    }

    pub fn clear(mut self) -> Result<Self, ENNError> {
        match self.backend {
            PosteriorBackend::Enn(enn) => {
                let inner = Arc::try_unwrap(enn).unwrap_or_else(|arc| (*arc).clone());
                let cleared = inner.cleared()?;
                self.backend = PosteriorBackend::Enn(Arc::new(cleared));
                Ok(self)
            }
            PosteriorBackend::External(ext) => {
                let boxed = match Arc::try_unwrap(ext) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                let cleared = boxed.cleared()?;
                self.backend = PosteriorBackend::External(Arc::new(cleared));
                Ok(self)
            }
            PosteriorBackend::Empirical { x_log, .. } => {
                let ncols = x_log.ncols();
                self.backend = PosteriorBackend::Empirical {
                    x_log: ImLog::with_cols(ncols),
                    y_log: ImLog::empty(),
                };
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
            PosteriorBackend::Enn(enn) => {
                let inner = Arc::try_unwrap(enn).unwrap_or_else(|arc| (*arc).clone());
                let thawed = inner.with_thawed(bytes, pos)?;
                self.backend = PosteriorBackend::Enn(Arc::new(thawed));
                Ok(self)
            }
            PosteriorBackend::External(ext) => {
                let boxed = match Arc::try_unwrap(ext) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                let thawed = boxed.with_thawed(bytes, pos)?;
                self.backend = PosteriorBackend::External(Arc::new(thawed));
                Ok(self)
            }
            PosteriorBackend::Empirical { .. } => self.clear(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            PosteriorBackend::Disk { .. } => self.clear(),
        }
    }

    pub fn schedule_flush(&mut self) -> Result<(), ENNError> {
        match &mut self.backend {
            PosteriorBackend::Enn(s) => {
                s.schedule_flush()?;
            }
            PosteriorBackend::External(s) => {
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
            PosteriorBackend::Enn(s) => {
                s.wait_flush()?;
            }
            PosteriorBackend::External(s) => {
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
            PosteriorBackend::Enn(s) => {
                s.release_pages()?;
            }
            PosteriorBackend::External(s) => {
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
        Ok(self.clone())
    }

    pub fn fork(&self) -> Result<Self, ENNError> {
        Ok(self.clone())
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
            PosteriorBackend::Enn(s) => {
                buf.push(2);
                let before = buf.len();
                <ENNSurrogate as Surrogate>::hash_surrogate(s.as_ref(), buf);
                if buf.len() == before {
                    hash_rows(buf);
                }
            }
            PosteriorBackend::External(s) => {
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

impl Default for PosteriorStore {
    fn default() -> Self {
        Self {
            backend: PosteriorBackend::Empirical {
                x_log: ImLog::empty(),
                y_log: ImLog::empty(),
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
