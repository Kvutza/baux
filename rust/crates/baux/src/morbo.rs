//! Morbo trust region: pure functional Chebyshev scalarization and arbitrary
//! multi-region hypervolume control over an inner TuRBO trust region.

use std::sync::Arc;

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::{Rng, RngCore};

use crate::hash::{AffineKey, Key};
#[cfg(test)]
use crate::trust_region::TRLengthConfig;
pub use crate::trust_region::{MorboTRSettings, Rescalarize};
use crate::trust_region::{TrustRegionError, TurboTrustRegion};

/// Pure deterministic Dirichlet(1) (`Exp(1)` simplex) projection from `(0, 1)`
/// uniforms.
pub fn simplex_uniforms(uniforms: &[f64]) -> Array1<f64> {
    let num_metrics = uniforms.len();
    if num_metrics == 0 {
        return Array1::zeros(0);
    }
    let mut out = Array1::zeros(num_metrics);
    let mut sum = 0.0;
    for (m, &raw) in uniforms.iter().enumerate() {
        let u = raw.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let g = (-u.ln()).max(1e-300);
        out[m] = g;
        sum += g;
    }
    if sum > 0.0 && sum.is_finite() {
        out.mapv_inplace(|v| v / sum);
    } else {
        out.fill(1.0 / num_metrics as f64);
    }
    out
}

/// Pure stateless Dirichlet(1) (`Exp(1)` simplex) draw derived from `Key::fold`
/// via shared `#![no_std]` [`baux::semantics::morbo::sample_simplex`].
pub fn sample_simplex(num_metrics: usize, key: Key) -> Array1<f64> {
    let mut out = Array1::zeros(num_metrics);
    baux::semantics::morbo::sample_simplex(key, out.as_slice_mut().expect("contiguous"));
    out
}

/// Pure stateless preference-weighted exponential simplex draw from
/// `Key::fold`.
pub fn sample_prefs(prefs: &[f64], key: Key) -> Result<Array1<f64>, TrustRegionError> {
    if prefs.is_empty() || prefs.iter().any(|&p| !p.is_finite() || p <= 0.0) {
        return Err(TrustRegionError::InvalidParameter(
            "MORBO preferences must contain positive finite values".into(),
        ));
    }
    let mut out = Array1::zeros(prefs.len());
    let mut sum = 0.0;
    for (m, &pref) in prefs.iter().enumerate() {
        let u = baux::semantics::random::u64_f53(key.fold(m as u64).seed())
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let g = (-u.ln() * pref).max(1e-300);
        out[m] = g;
        sum += g;
    }
    if sum > 0.0 && sum.is_finite() {
        out.mapv_inplace(|v| v / sum);
    } else {
        let p_sum: f64 = prefs.iter().sum();
        for (m, &pref) in prefs.iter().enumerate() {
            out[m] = pref / p_sum;
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct MorboTrustRegion {
    inner: TurboTrustRegion,
    num_metrics: usize,
    alpha: f64,
    rescalarize: Rescalarize,
    noise_aware: bool,
    weights: Arc<Array1<f64>>,
    y_min: Option<Arc<Array1<f64>>>,
    y_max: Option<Arc<Array1<f64>>>,
    incumbent_y_raw: Option<Arc<Array1<f64>>>,
    score_min: Option<f64>,
    score_max: Option<f64>,
    score_count: usize,
}

impl MorboTrustRegion {
    /// Pure functional constructor driven by an immutable [`Key`].
    pub fn new_key(
        num_dim: usize,
        settings: MorboTRSettings,
        key: Key,
    ) -> Result<Self, TrustRegionError> {
        if settings.num_metrics < 2 {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_metrics must be >= 2 for MORBO, got {}",
                settings.num_metrics
            )));
        }
        let inner = TurboTrustRegion::new(num_dim, settings.length);
        let weights = Arc::new(sample_simplex(settings.num_metrics, key));
        Ok(Self {
            inner,
            num_metrics: settings.num_metrics,
            alpha: settings.alpha,
            rescalarize: settings.rescalarize,
            noise_aware: settings.noise_aware,
            weights,
            y_min: None,
            y_max: None,
            incumbent_y_raw: None,
            score_min: None,
            score_max: None,
            score_count: 0,
        })
    }

    /// Affine-key constructor advancing the PRNG token in a single linear step.
    pub fn new_affine(
        num_dim: usize,
        settings: MorboTRSettings,
        key: AffineKey,
    ) -> Result<(Self, AffineKey), TrustRegionError> {
        let (draw, next) = key.step();
        Ok((Self::new_key(num_dim, settings, draw)?, next))
    }

    /// Backwards-compatible adapter delegating to pure [`simplex_uniforms`].
    pub fn new(
        num_dim: usize,
        settings: MorboTRSettings,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        let base = Self::new_key(num_dim, settings, Key::default())?;
        Ok(base.resample_weights(rng))
    }

    #[inline(always)]
    pub const fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    #[inline(always)]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline(always)]
    pub const fn noise_aware(&self) -> bool {
        self.noise_aware
    }

    #[inline(always)]
    pub const fn length(&self) -> f64 {
        self.inner.length()
    }

    #[inline(always)]
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn set_weights(mut self, weights: &[f64]) -> Result<Self, TrustRegionError> {
        if weights.len() != self.num_metrics
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(TrustRegionError::InvalidParameter(
                "MORBO weights must contain one positive finite value per metric".into(),
            ));
        }
        let sum = weights.iter().sum::<f64>();
        self.weights = Arc::new(Array1::from_iter(weights.iter().map(|weight| weight / sum)));
        Ok(self)
    }

    pub fn with_weights(self, weights: &[f64]) -> Result<Self, TrustRegionError> {
        self.set_weights(weights)
    }

    pub fn y_min(&self) -> Option<&Array1<f64>> {
        self.y_min.as_deref()
    }

    pub fn y_max(&self) -> Option<&Array1<f64>> {
        self.y_max.as_deref()
    }

    #[inline(always)]
    pub const fn rescalarize(&self) -> Rescalarize {
        self.rescalarize
    }

    /// Pure functional weight resampling from an immutable [`Key`].
    pub fn resample_key(mut self, key: Key) -> Self {
        self.weights = Arc::new(sample_simplex(self.num_metrics, key));
        self
    }

    /// Affine-key weight resampling returning `(updated_tr, next_key)`.
    pub fn resample_affine(self, key: AffineKey) -> (Self, AffineKey) {
        let (draw, next) = key.step();
        (self.resample_key(draw), next)
    }

    /// Backwards-compatible adapter delegating to pure [`simplex_uniforms`].
    pub fn resample_weights(mut self, rng: &mut dyn RngCore) -> Self {
        let uniforms: Vec<f64> = (0..self.num_metrics).map(|_| rng.r#gen::<f64>()).collect();
        self.weights = Arc::new(simplex_uniforms(&uniforms));
        self
    }

    /// Pure proposal-time rescalarization gate supporting both
    /// [`Rescalarize::OnRestart`] and [`Rescalarize::OnPropose`]
    /// (`Rescalarize::EveryStep`).
    pub fn resample_propose(self, key: Key) -> (Self, bool) {
        if self.rescalarize == Rescalarize::OnPropose {
            (self.resample_key(key), true)
        } else {
            (self, false)
        }
    }

    /// Atomic pure proposal transition applying per-step rescalarization and
    /// incumbent baseline updates when `Rescalarize::OnPropose` (`EveryStep`)
    /// is configured.
    pub fn step_propose(
        self,
        key: Key,
        num_obs: usize,
        incumbent: Option<Array1<f64>>,
    ) -> Result<Self, TrustRegionError> {
        let (next, changed) = self.resample_propose(key);
        if changed {
            next.rescalarize_weights(num_obs, incumbent)
        } else {
            Ok(next)
        }
    }

    pub fn set_arms(mut self, num_arms: usize) -> Self {
        self.inner = self.inner.with_arms(num_arms);
        self
    }

    pub fn with_arms(self, num_arms: usize) -> Self {
        self.set_arms(num_arms)
    }

    pub fn set_tolerance(mut self, failures: usize) -> Result<Self, TrustRegionError> {
        self.inner = self.inner.with_tolerance(failures)?;
        Ok(self)
    }

    pub fn with_tolerance(self, failures: usize) -> Result<Self, TrustRegionError> {
        self.set_tolerance(failures)
    }

    pub fn success_tolerance(&self) -> i32 {
        self.inner.success_tolerance()
    }

    pub fn failure_tolerance(&self) -> i32 {
        self.inner.failure_tolerance()
    }

    pub fn success_counter(&self) -> i32 {
        self.inner.success_counter()
    }

    pub fn failure_counter(&self) -> i32 {
        self.inner.failure_counter()
    }

    pub fn needs_restart(&self) -> bool {
        self.inner.needs_restart()
    }

    /// Pure functional restart accepting an optional [`Key`] for
    /// [`Rescalarize::OnRestart`].
    pub fn restart_key(mut self, key: Option<Key>) -> Self {
        self.y_min = None;
        self.y_max = None;
        self.incumbent_y_raw = None;
        self.score_min = None;
        self.score_max = None;
        self.score_count = 0;
        self.inner = self.inner.restarted_local();
        if let Some(k) = key {
            if self.rescalarize == Rescalarize::OnRestart {
                self = self.resample_key(k);
            }
        }
        self
    }

    /// Backwards-compatible adapter delegating to [`Self::restart_key`] +
    /// [`Self::resample_weights`].
    pub fn restart(mut self, rng: Option<&mut dyn RngCore>) -> Self {
        self = self.restart_key(None);
        if let Some(r) = rng
            && self.rescalarize == Rescalarize::OnRestart
        {
            self = self.resample_weights(r);
        }
        self
    }

    pub fn with_restart(self, rng: Option<&mut dyn RngCore>) -> Self {
        self.restart(rng)
    }

    pub(crate) fn freeze_morbo(&self, buf: &mut Vec<u8>) {
        use crate::snapshot_io::{write_arr, write_bool, write_len, write_real};
        self.inner.freeze_turbo(buf);
        write_len(buf, self.num_metrics);
        write_arr(buf, &self.weights);
        match &self.y_min {
            Some(v) => {
                write_bool(buf, true);
                write_arr(buf, v);
            }
            None => write_bool(buf, false),
        }
        match &self.y_max {
            Some(v) => {
                write_bool(buf, true);
                write_arr(buf, v);
            }
            None => write_bool(buf, false),
        }
        match &self.incumbent_y_raw {
            Some(v) => {
                write_bool(buf, true);
                write_arr(buf, v);
            }
            None => write_bool(buf, false),
        }
        match self.score_min {
            Some(v) => {
                write_bool(buf, true);
                write_real(buf, v);
            }
            None => write_bool(buf, false),
        }
        match self.score_max {
            Some(v) => {
                write_bool(buf, true);
                write_real(buf, v);
            }
            None => write_bool(buf, false),
        }
        write_len(buf, self.score_count);
    }

    pub(crate) fn thaw_morbo(
        mut self,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, crate::error::ENNError> {
        use crate::snapshot_io::{MAX_METRICS, MAX_OBS, bad_state};
        self.inner.thaw_turbo(reader)?;
        let nm = reader.read_len(MAX_METRICS)?;
        if nm != self.num_metrics {
            return Err(bad_state("frozen MORBO metric count mismatch"));
        }
        let weights = reader.read_arr(nm, true)?;
        if weights.iter().any(|&w| w <= 0.0) {
            return Err(bad_state("frozen MORBO weights must be positive"));
        }
        let y_min = if reader.read_bool()? {
            Some(Arc::new(reader.read_arr(nm, true)?))
        } else {
            None
        };
        let y_max = if reader.read_bool()? {
            Some(Arc::new(reader.read_arr(nm, true)?))
        } else {
            None
        };
        if y_min.is_some() != y_max.is_some() {
            return Err(bad_state("frozen MORBO y_min/y_max presence mismatch"));
        }
        let incumbent_y_raw = if reader.read_bool()? {
            Some(Arc::new(reader.read_arr(nm, true)?))
        } else {
            None
        };
        let score_min = if reader.read_bool()? {
            Some(reader.read_finite()?)
        } else {
            None
        };
        let score_max = if reader.read_bool()? {
            Some(reader.read_finite()?)
        } else {
            None
        };
        let score_count = reader.read_len(MAX_OBS)?;
        self.weights = Arc::new(weights);
        self.y_min = y_min;
        self.y_max = y_max;
        self.incumbent_y_raw = incumbent_y_raw;
        self.score_min = score_min;
        self.score_max = score_max;
        self.score_count = score_count;
        Ok(self)
    }

    pub(crate) fn hash_morbo(&self, buf: &mut Vec<u8>) {
        self.freeze_morbo(buf);
    }

    /// Zero-allocation scalar `(lower, upper)` bound query for one coordinate.
    #[inline(always)]
    pub fn bound_dim(&self, center: f64, scale: Option<f64>) -> (f64, f64) {
        self.inner.bound_dim(center, scale)
    }

    pub fn compute_bounds(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        self.inner.compute_bounds(x_center, lengthscales)
    }

    /// Zero-allocation Chebyshev scalarization of a single objective slice.
    pub fn scalarize_row(&self, row: &[f64], clip: bool) -> Result<f64, TrustRegionError> {
        if row.len() != self.num_metrics {
            return Err(TrustRegionError::InvalidParameter(format!(
                "row len {} != num_metrics {}",
                row.len(),
                self.num_metrics
            )));
        }
        let (y_min, y_max) = self
            .y_min
            .as_deref()
            .zip(self.y_max.as_deref())
            .ok_or_else(|| {
                TrustRegionError::InvalidState("scalarize called before observations".to_string())
            })?;
        Ok(chebyshev_point(
            row,
            y_min.as_slice().expect("contiguous y_min"),
            y_max.as_slice().expect("contiguous y_max"),
            self.weights.as_slice().expect("contiguous weights"),
            self.alpha,
            clip,
        ))
    }

    pub fn scalarize(
        &self,
        y: &ArrayView2<f64>,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        let (y_min, y_max) = self
            .y_min
            .as_deref()
            .zip(self.y_max.as_deref())
            .ok_or_else(|| {
                TrustRegionError::InvalidState("scalarize called before observations".to_string())
            })?;
        scalarize_ranges(
            y,
            &y_min.view(),
            &y_max.view(),
            &self.weights.view(),
            self.alpha,
            self.num_metrics,
            clip,
        )
    }

    /// Noise-aware Chebyshev scalarization shifting `mu` by `beta * sqrt(yvar)`
    /// when heteroscedastic observation variances are supplied.
    pub fn scalarize_noisy(
        &self,
        mu: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        beta: f64,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        let Some(yv) = yvar.filter(|_| self.noise_aware || beta != 0.0) else {
            return self.scalarize(mu, clip);
        };
        if mu.shape() != yv.shape() || mu.ncols() != self.num_metrics {
            return Err(TrustRegionError::InvalidParameter(format!(
                "shape mismatch in scalarize_noisy: mu {:?}, yvar {:?}",
                mu.shape(),
                yv.shape()
            )));
        }
        let (y_min, y_max) = self
            .y_min
            .as_deref()
            .zip(self.y_max.as_deref())
            .ok_or_else(|| {
                TrustRegionError::InvalidState("scalarize called before observations".to_string())
            })?;
        let min_s = y_min.as_slice().expect("contiguous y_min");
        let max_s = y_max.as_slice().expect("contiguous y_max");
        let w_s = self.weights.as_slice().expect("contiguous weights");
        let n = mu.nrows();
        let mut scores = Array1::zeros(n);
        let mut shifted = vec![0.0; self.num_metrics];
        for row in 0..n {
            for m in 0..self.num_metrics {
                let var = yv[[row, m]];
                shifted[m] = if var.is_finite() && var >= 0.0 {
                    mu[[row, m]] + beta * var.sqrt()
                } else {
                    f64::NAN
                };
            }
            scores[row] = chebyshev_point(&shifted, min_s, max_s, w_s, self.alpha, clip);
        }
        Ok(scores)
    }

    /// Chebyshev scores using the minimum and maximum of `y` itself.
    pub fn scalarize_local(&self, y: &ArrayView2<f64>) -> Result<Array1<f64>, TrustRegionError> {
        let m = y.ncols();
        let mut y_min = Array1::from_elem(m, f64::INFINITY);
        let mut y_max = Array1::from_elem(m, f64::NEG_INFINITY);
        for i in 0..y.nrows() {
            for j in 0..m {
                let v = y[[i, j]];
                if v.is_finite() {
                    y_min[j] = y_min[j].min(v);
                    y_max[j] = y_max[j].max(v);
                }
            }
        }
        for j in 0..m {
            if !y_min[j].is_finite() || !y_max[j].is_finite() {
                y_min[j] = 0.0;
                y_max[j] = 0.0;
            }
        }
        scalarize_ranges(
            y,
            &y_min.view(),
            &y_max.view(),
            &self.weights.view(),
            self.alpha,
            self.num_metrics,
            true,
        )
    }

    /// FBIP incremental range update using `Arc::make_mut` when unshared.
    pub fn update_incremental(mut self, y_new: &ArrayView2<f64>) -> Self {
        if y_new.nrows() == 0 {
            return self;
        }
        if self.y_min.is_none() || self.y_max.is_none() {
            let mut y_min = Array1::from_elem(self.num_metrics, f64::INFINITY);
            let mut y_max = Array1::from_elem(self.num_metrics, f64::NEG_INFINITY);
            for row in 0..y_new.nrows() {
                for m in 0..self.num_metrics {
                    let v = y_new[[row, m]];
                    if v.is_finite() {
                        y_min[m] = y_min[m].min(v);
                        y_max[m] = y_max[m].max(v);
                    }
                }
            }
            self.y_min = Some(Arc::new(y_min));
            self.y_max = Some(Arc::new(y_max));
            return self;
        }
        let y_min = Arc::make_mut(self.y_min.as_mut().expect("y_min"));
        let y_max = Arc::make_mut(self.y_max.as_mut().expect("y_max"));
        for row in 0..y_new.nrows() {
            for m in 0..self.num_metrics {
                let v = y_new[[row, m]];
                if v.is_finite() {
                    y_min[m] = y_min[m].min(v);
                    y_max[m] = y_max[m].max(v);
                }
            }
        }
        self
    }

    pub fn with_incremental(self, y_new: &ArrayView2<f64>) -> Self {
        self.update_incremental(y_new)
    }

    pub fn update_only(
        mut self,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, TrustRegionError> {
        if y_incumbent.len() != self.num_metrics {
            return Err(TrustRegionError::InvalidParameter(format!(
                "y_incumbent len {} != num_metrics {}",
                y_incumbent.len(),
                self.num_metrics
            )));
        }
        if num_obs == 0 {
            self.y_min = None;
            self.y_max = None;
            self.incumbent_y_raw = None;
            self.inner = self.inner.restarted();
            return Ok(self);
        }
        let (min_arr, max_arr) = self
            .y_min
            .as_deref()
            .zip(self.y_max.as_deref())
            .ok_or_else(|| {
                TrustRegionError::InvalidState("update_only before ranges initialized".to_string())
            })?;
        let min_s = min_arr.as_slice().expect("contiguous y_min");
        let max_s = max_arr.as_slice().expect("contiguous y_max");
        let w_s = self.weights.as_slice().expect("contiguous weights");

        if self.incumbent_y_raw.is_none() {
            let owned = y_incumbent.to_owned();
            let score = chebyshev_point(
                owned.as_slice().expect("contiguous"),
                min_s,
                max_s,
                w_s,
                self.alpha,
                true,
            );
            self.incumbent_y_raw = Some(Arc::new(owned));
            return push_score(self, num_obs, score);
        }

        let old_score = chebyshev_point(
            self.incumbent_y_raw
                .as_deref()
                .expect("incumbent")
                .as_slice()
                .expect("contiguous"),
            min_s,
            max_s,
            w_s,
            self.alpha,
            true,
        );
        let new_score = chebyshev_view(y_incumbent, min_s, max_s, w_s, self.alpha, true);
        self.inner = self.inner.with_best(old_score);
        self = push_score(self, num_obs, new_score)?;
        if new_score > old_score {
            self.incumbent_y_raw = Some(Arc::new(y_incumbent.to_owned()));
        }
        Ok(self)
    }

    pub fn rescalarize_weights(
        mut self,
        num_obs: usize,
        incumbent: Option<Array1<f64>>,
    ) -> Result<Self, TrustRegionError> {
        if let Some(y_inc) = incumbent {
            if y_inc.len() == self.num_metrics && y_inc.iter().all(|v| v.is_finite()) {
                self.incumbent_y_raw = Some(Arc::new(y_inc));
            }
        }
        let Some(y_inc) = self.incumbent_y_raw.as_deref() else {
            return Ok(self);
        };
        let (Some(y_min), Some(y_max)) = (self.y_min.as_deref(), self.y_max.as_deref()) else {
            return Ok(self);
        };
        let score = chebyshev_point(
            y_inc.as_slice().expect("contiguous"),
            y_min.as_slice().expect("contiguous"),
            y_max.as_slice().expect("contiguous"),
            self.weights.as_slice().expect("contiguous"),
            self.alpha,
            true,
        );
        self.inner = self.inner.with_best(score);
        if num_obs > self.inner.prev_obs() {
            self = push_score(self, num_obs, score)?;
        }
        Ok(self)
    }

    pub fn update(
        mut self,
        y_obs: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
    ) -> Result<Self, TrustRegionError> {
        let n = y_obs.nrows();
        if y_obs.ncols() != self.num_metrics {
            return Err(TrustRegionError::InvalidParameter(format!(
                "y_obs cols {} != num_metrics {}",
                y_obs.ncols(),
                self.num_metrics
            )));
        }
        if y_incumbent.len() != self.num_metrics {
            return Err(TrustRegionError::InvalidParameter(format!(
                "y_incumbent len {} != num_metrics {}",
                y_incumbent.len(),
                self.num_metrics
            )));
        }
        if n == 0 {
            self.y_min = None;
            self.y_max = None;
            self.incumbent_y_raw = None;
            self.inner = self.inner.restarted();
            return Ok(self);
        }
        let prev_n = self.inner.prev_obs();
        if n < prev_n {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                n, prev_n
            )));
        }
        let mut y_min = Array1::from_elem(self.num_metrics, f64::INFINITY);
        let mut y_max = Array1::from_elem(self.num_metrics, f64::NEG_INFINITY);
        for m in 0..self.num_metrics {
            for row in 0..n {
                let v = y_obs[[row, m]];
                if v.is_finite() {
                    y_min[m] = y_min[m].min(v);
                    y_max[m] = y_max[m].max(v);
                }
            }
        }
        self.y_min = Some(Arc::new(y_min));
        self.y_max = Some(Arc::new(y_max));

        let min_s = self.y_min.as_deref().unwrap().as_slice().unwrap();
        let max_s = self.y_max.as_deref().unwrap().as_slice().unwrap();
        let w_s = self.weights.as_slice().unwrap();

        if prev_n == 0 || self.incumbent_y_raw.is_none() {
            let owned = y_incumbent.to_owned();
            let score = chebyshev_point(
                owned.as_slice().unwrap(),
                min_s,
                max_s,
                w_s,
                self.alpha,
                true,
            );
            self.incumbent_y_raw = Some(Arc::new(owned));
            return push_score(self, n, score);
        }

        let old_score = chebyshev_point(
            self.incumbent_y_raw.as_deref().unwrap().as_slice().unwrap(),
            min_s,
            max_s,
            w_s,
            self.alpha,
            true,
        );
        let new_score = chebyshev_view(y_incumbent, min_s, max_s, w_s, self.alpha, true);
        self.inner = self.inner.with_best(old_score);
        self = push_score(self, n, new_score)?;
        if new_score > old_score {
            self.incumbent_y_raw = Some(Arc::new(y_incumbent.to_owned()));
        }
        Ok(self)
    }

    pub fn with_update(
        self,
        y_obs: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
    ) -> Result<Self, TrustRegionError> {
        self.update(y_obs, y_incumbent)
    }

    pub fn update_batch(
        self,
        y_obs: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
    ) -> Result<Self, TrustRegionError> {
        self.update(y_obs, y_incumbent)
    }
}

/// Select the active MORBO trust-region slot over an arbitrary region count
/// (`1..=usize::MAX`) via exact exclusive hypervolume UCB1.
pub fn select_region(
    trials: &[u64],
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<usize, TrustRegionError> {
    if trials.is_empty() || trials.len() != y.nrows() {
        return Err(TrustRegionError::InvalidParameter(
            "MORBO region trials must be non-empty and match incumbent rows".into(),
        ));
    }
    if let Some(untried) = trials.iter().position(|&t| t == 0) {
        return Ok(untried);
    }
    let contrib = crate::hypervolume::hv_contributions(y, ref_point)
        .map_err(|e| TrustRegionError::InvalidParameter(e.to_string()))?;
    let total_trials = (trials.iter().sum::<u64>() as f64).max(1.0);
    Ok((0..trials.len())
        .max_by(|&left, &right| {
            let l_score = contrib[left] + (2.0 * total_trials.ln() / trials[left] as f64).sqrt();
            let r_score = contrib[right] + (2.0 * total_trials.ln() / trials[right] as f64).sqrt();
            l_score.total_cmp(&r_score).then_with(|| right.cmp(&left))
        })
        .unwrap_or(0))
}

/// Select the restart source slot maximizing exclusive hypervolume over an
/// arbitrary number of MORBO regions (`1..=usize::MAX`).
pub fn max_region(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
    excluded: Option<usize>,
) -> Result<usize, TrustRegionError> {
    if y.nrows() == 0 {
        return Err(TrustRegionError::InvalidParameter(
            "MORBO max_region requires at least one region".into(),
        ));
    }
    let contrib = crate::hypervolume::hv_contributions(y, ref_point)
        .map_err(|e| TrustRegionError::InvalidParameter(e.to_string()))?;
    (0..y.nrows())
        .filter(|&idx| Some(idx) != excluded)
        .max_by(|&left, &right| {
            contrib[left]
                .total_cmp(&contrib[right])
                .then_with(|| right.cmp(&left))
        })
        .ok_or_else(|| {
            TrustRegionError::InvalidParameter("no eligible MORBO region remaining".into())
        })
}

fn push_score(
    mut morbo: MorboTrustRegion,
    num_obs: usize,
    score: f64,
) -> Result<MorboTrustRegion, TrustRegionError> {
    let scale = if morbo.score_count >= 2 {
        match (morbo.score_min, morbo.score_max) {
            (Some(low), Some(high)) => (high - low).max(1e-6),
            _ => 0.0,
        }
    } else {
        0.0
    };
    morbo.inner = morbo.inner.step_scalar(num_obs, score, scale)?;
    morbo.score_min = Some(
        morbo
            .score_min
            .map(|value| value.min(score))
            .unwrap_or(score),
    );
    morbo.score_max = Some(
        morbo
            .score_max
            .map(|value| value.max(score))
            .unwrap_or(score),
    );
    morbo.score_count += 1;
    Ok(morbo)
}

#[inline]
fn chebyshev_point(
    row: &[f64],
    y_min: &[f64],
    y_max: &[f64],
    weights: &[f64],
    alpha: f64,
    clip: bool,
) -> f64 {
    baux::semantics::morbo::chebyshev(row, y_min, y_max, weights, alpha, clip)
}

#[inline]
fn chebyshev_view(
    row: &ArrayView1<f64>,
    y_min: &[f64],
    y_max: &[f64],
    weights: &[f64],
    alpha: f64,
    clip: bool,
) -> f64 {
    if let Some(slice) = row.as_slice() {
        return chebyshev_point(slice, y_min, y_max, weights, alpha, clip);
    }
    let buf: Vec<f64> = row.iter().copied().collect();
    chebyshev_point(&buf, y_min, y_max, weights, alpha, clip)
}

pub fn scalarize_ranges(
    y: &ArrayView2<f64>,
    y_min: &ArrayView1<f64>,
    y_max: &ArrayView1<f64>,
    weights: &ArrayView1<f64>,
    alpha: f64,
    num_metrics: usize,
    clip: bool,
) -> Result<Array1<f64>, TrustRegionError> {
    if y.ncols() != num_metrics
        || y_min.len() != num_metrics
        || y_max.len() != num_metrics
        || weights.len() != num_metrics
    {
        return Err(TrustRegionError::InvalidParameter(format!(
            "shape mismatch: y {:?}, y_min {}, y_max {}, weights {}",
            y.shape(),
            y_min.len(),
            y_max.len(),
            weights.len()
        )));
    }
    let n = y.nrows();
    let mut scores = Array1::zeros(n);
    let min_std = y_min.as_standard_layout();
    let max_std = y_max.as_standard_layout();
    let w_std = weights.as_standard_layout();
    let min_s = min_std.as_slice().expect("standard layout");
    let max_s = max_std.as_slice().expect("standard layout");
    let w_s = w_std.as_slice().expect("standard layout");
    if let Some(y_slice) = y.as_slice() {
        baux::semantics::morbo::scalarize_slice(
            y_slice,
            n,
            num_metrics,
            min_s,
            max_s,
            w_s,
            alpha,
            clip,
            scores.as_slice_mut().expect("contiguous"),
        );
    } else {
        for r in 0..n {
            scores[r] = chebyshev_view(&y.row(r), min_s, max_s, w_s, alpha, clip);
        }
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn morbo_range() {
        let y = array![[1.0, 2.0], [1.0, 3.0]];
        let y_min = array![1.0, 2.0];
        let y_max = array![1.0, 5.0];
        let w = array![0.5, 0.5];
        let scores = scalarize_ranges(
            &y.view(),
            &y_min.view(),
            &y_max.view(),
            &w.view(),
            0.1,
            2,
            true,
        )
        .unwrap();
        assert!((scores[0] - 0.025).abs() < 1e-12);
        assert!((scores[1] - 0.208_333_333_333_333_34).abs() < 1e-12);
    }

    #[test]
    fn incumbent_ranges() {
        let settings = MorboTRSettings {
            num_metrics: 2,
            alpha: 0.1,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let tr = MorboTrustRegion::new_key(2, settings, Key::new(7))
            .unwrap()
            .set_weights(&[0.5, 0.5])
            .unwrap()
            .update_incremental(&array![[0.0, 0.0], [100.0, 10.0]].view());

        let candidates = array![[10.0, 9.0], [9.0, 10.0]];
        let local = tr.scalarize_local(&candidates.view()).unwrap();
        let global = tr.scalarize(&candidates.view(), true).unwrap();
        let row0 = tr.scalarize_row(&[10.0, 9.0], true).unwrap();

        assert!((local[0] - 0.05).abs() < 1e-12);
        assert!((local[1] - 0.05).abs() < 1e-12);
        assert!((row0 - global[0]).abs() < 1e-12);
        assert_ne!(local, global);
    }

    #[test]
    fn morbo_2026() {
        let settings = MorboTRSettings {
            num_metrics: 3,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let mut rng = StdRng::seed_from_u64(2026);
        let tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
        let s: f64 = tr.weights().iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        assert!(tr.weights().iter().all(|&x| x > 0.0));
    }

    #[test]
    fn morbo_metrics2() {
        let settings = MorboTRSettings {
            num_metrics: 1,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let result = MorboTrustRegion::new_key(2, settings, Key::new(7));
        assert!(
            result.is_err(),
            "MORBO requires num_metrics >= 2 (Python MultiObjectiveConfig parity)"
        );
    }

    #[test]
    fn morbo_panic() {
        let settings = MorboTRSettings {
            num_metrics: 0,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let result = MorboTrustRegion::new_key(2, settings, Key::new(99));
        assert!(result.is_err());
    }

    #[test]
    fn morbo_initial() {
        let settings = MorboTRSettings {
            num_metrics: 2,
            alpha: 0.1,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let mut tr = MorboTrustRegion::new_key(2, settings, Key::new(1))
            .unwrap()
            .set_arms(2);
        let y = array![[1.0, 2.0], [2.0, 3.0]];
        let inc = array![2.0, 3.0];
        tr = tr.update(&y.view(), &inc.view()).unwrap();
        assert!(tr.incumbent_y_raw.is_some());
        assert_eq!(tr.score_count, 1);
        assert_eq!(tr.score_min, tr.score_max);

        tr = tr.restart_key(None);
        assert_eq!(tr.score_count, 0);
        assert_eq!(tr.score_min, None);
        assert_eq!(tr.score_max, None);
    }

    #[test]
    fn batch_update() {
        let settings = MorboTRSettings {
            num_metrics: 2,
            alpha: 0.1,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
        let y = array![[1.0, 2.0], [2.0, 3.0]];
        let inc = array![2.0, 3.0];
        tr = tr.update_batch(&y.view(), &inc.view()).unwrap();
        assert!(tr.incumbent_y_raw.is_some());
        assert_eq!(tr.score_count, 1);
    }

    #[test]
    fn key_arbitraries() {
        let settings = MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::EveryStep,
            noise_aware: true,
        };
        let (tr, ak) = MorboTrustRegion::new_affine(3, settings, AffineKey::new(123)).unwrap();
        assert!(tr.noise_aware());
        let fork = tr.clone();
        assert!(Arc::ptr_eq(&tr.weights, &fork.weights));

        let y = array![[1.0, 2.0], [3.0, 1.0]];
        let yvar = array![[0.01, 0.25], [0.25, 0.01]];
        let tr = tr.update_incremental(&y.view());
        let noisy = tr
            .scalarize_noisy(&y.view(), Some(&yvar.view()), -1.0, true)
            .unwrap();
        let noiseless = tr.scalarize(&y.view(), true).unwrap();
        assert!(noisy[0] < noiseless[0] && noisy[1] < noiseless[1]);

        let w0 = tr.weights().clone();
        let tr = tr
            .step_propose(ak.into_key(), 2, Some(array![3.0, 1.0]))
            .unwrap();
        assert_ne!(tr.weights(), &w0);
        assert_eq!(fork.weights(), &w0);

        // Arbitrary 16-region MORBO portfolio selection (beyond 2..=8).
        let mut y16 = ndarray::Array2::zeros((16, 2));
        for r in 0..16 {
            y16[[r, 0]] = (r + 1) as f64;
            y16[[r, 1]] = (16 - r) as f64;
        }
        let trials = vec![2u64; 16];
        let ref_pt = array![0.0, 0.0];
        let chosen = select_region(&trials, &y16.view(), &ref_pt.view()).unwrap();
        assert!(chosen < 16);
        let best_src = max_region(&y16.view(), &ref_pt.view(), Some(chosen)).unwrap();
        assert!(best_src < 16 && best_src != chosen);
    }
}
