//! Stateful ENN hyperparameter fitting with incremental statistics.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::RngCore;

use crate::error::ENNError;
use crate::hash::{AffineKey, Key};
use crate::model::ENN;
use crate::params::ENNParams;

/// Pure functional hyperparameter fitting over an immutable `ENN` model.
pub fn fit_model(
    model: &ENN,
    k: i32,
    num_candidates: usize,
    num_samples: usize,
    infer_aleatoric: bool,
    warm: Option<ENNParams>,
    key: Key,
) -> Result<ENNParams, ENNError> {
    let mut fitter = ENNFitter::new(k, infer_aleatoric);
    if model.bounded_outputs() {
        fitter = fitter.with_shared(Some(model.shared_bounds()));
    }
    let (_fitter, params) =
        fitter.ask_key(model, num_candidates, num_samples, warm.as_ref(), key)?;
    Ok(params)
}

/// FBIP persistent ENN fitter: `Arc`-shared running `y` moments and warm-start
/// params.
#[derive(Debug, Clone)]
pub struct ENNFitter {
    k: i32,
    infer_aleatoric_variance: bool,
    params: Option<ENNParams>,
    y_bounds: Option<Arc<Array2<f64>>>,
    y_sum: Arc<Array1<f64>>,
    y_sumsq: Arc<Array1<f64>>,
    y_count: usize,
    num_metrics: usize,
}

impl ENNFitter {
    pub fn new(k: i32, infer_aleatoric_variance: bool) -> Self {
        Self {
            k,
            infer_aleatoric_variance,
            params: None,
            y_bounds: None,
            y_sum: Arc::new(Array1::zeros(0)),
            y_sumsq: Arc::new(Array1::zeros(0)),
            y_count: 0,
            num_metrics: 0,
        }
    }

    pub fn with_bounds(mut self, bounds: Option<Array2<f64>>) -> Self {
        self.y_bounds = bounds.map(Arc::new);
        self
    }

    pub fn with_shared(mut self, bounds: Option<Arc<Array2<f64>>>) -> Self {
        self.y_bounds = bounds;
        self
    }

    pub fn set_bounds(&mut self, bounds: Option<Array2<f64>>) {
        self.y_bounds = bounds.map(Arc::new);
    }

    pub fn with_params(mut self, params: ENNParams) -> Self {
        self.params = Some(params);
        self
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.params.as_ref()
    }

    pub fn set_params(&mut self, params: ENNParams) {
        self.params = Some(params);
    }

    pub(crate) fn freeze_fitter(&self, buf: &mut Vec<u8>) {
        use crate::snapshot_io::{write_arr, write_bool, write_int, write_len, write_real};
        write_int(buf, self.k);
        write_bool(buf, self.infer_aleatoric_variance);
        match self.params {
            Some(p) => {
                write_bool(buf, true);
                write_int(buf, p.k_neighbors);
                write_real(buf, p.epistemic_scale);
                write_real(buf, p.aleatoric_scale);
            }
            None => write_bool(buf, false),
        }
        write_len(buf, self.y_count);
        write_len(buf, self.num_metrics);
        write_arr(buf, &self.y_sum);
        write_arr(buf, &self.y_sumsq);
    }

    pub(crate) fn thaw_fitter(
        y_bounds: Option<Array2<f64>>,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, ENNError> {
        use crate::snapshot_io::{MAX_METRICS, MAX_OBS, bad_state};
        let k = reader.read_int()?;
        if k <= 0 {
            return Err(bad_state("frozen fitter k must be > 0"));
        }
        let infer_aleatoric = reader.read_bool()?;
        let params = if reader.read_bool()? {
            let kn = reader.read_int()?;
            let ep = reader.read_finite()?;
            let al = reader.read_finite()?;
            Some(ENNParams::new(kn, ep, al).map_err(|e| bad_state(e.to_string()))?)
        } else {
            None
        };
        let y_count = reader.read_len(MAX_OBS)?;
        let num_metrics = reader.read_len(MAX_METRICS)?;
        let expected = if y_count == 0 { 0 } else { num_metrics };
        let y_sum = reader.read_arr(expected, true)?;
        let y_sumsq = reader.read_arr(expected, true)?;
        Ok(Self {
            k,
            infer_aleatoric_variance: infer_aleatoric,
            params,
            y_bounds: y_bounds.map(Arc::new),
            y_sum: Arc::new(y_sum),
            y_sumsq: Arc::new(y_sumsq),
            y_count,
            num_metrics,
        })
    }

    pub(crate) fn hash_fitter(&self, buf: &mut Vec<u8>) {
        self.freeze_fitter(buf);
    }

    pub fn y_stats(&mut self, y: &ArrayView2<f64>) {
        self.y_count = y.nrows();
        self.num_metrics = y.ncols();
        self.y_sum = Arc::new(y.sum_axis(Axis(0)));
        self.y_sumsq = Arc::new(y.mapv(|v| v * v).sum_axis(Axis(0)));
    }

    pub fn update_y(&mut self, y_new: &ArrayView2<f64>) {
        if y_new.nrows() == 0 {
            return;
        }
        self.num_metrics = y_new.ncols();
        if self.y_count == 0 {
            self.y_stats(y_new);
            return;
        }
        let y_sum = Arc::make_mut(&mut self.y_sum);
        let y_sumsq = Arc::make_mut(&mut self.y_sumsq);
        for row in y_new.axis_iter(Axis(0)) {
            for m in 0..self.num_metrics {
                let v = row[m];
                y_sum[m] += v;
                y_sumsq[m] += v * v;
            }
            self.y_count += 1;
        }
    }

    pub fn observe_y(mut self, y_new: &ArrayView2<f64>) -> Self {
        self.update_y(y_new);
        self
    }

    pub fn tell(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "x must contain only finite values".to_string(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "y must contain only finite values".to_string(),
            ));
        }
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "x and y must have same number of rows: {} vs {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if let Some(yv) = yvar {
            if yv.iter().any(|&v| !v.is_finite() || v < 0.0) {
                return Err(ENNError::InvalidParameter(
                    "yvar must contain only finite non-negative values".to_string(),
                ));
            }
            if yv.nrows() != y.nrows() {
                return Err(ENNError::InvalidParameter(format!(
                    "yvar and y must have same number of rows: {} vs {}",
                    yv.nrows(),
                    y.nrows()
                )));
            }
            if yv.ncols() != y.ncols() {
                return Err(ENNError::InvalidParameter(format!(
                    "yvar and y must have same number of columns: {} vs {}",
                    yv.ncols(),
                    y.ncols()
                )));
            }
        }
        if let Some(bounds) = self.y_bounds.as_deref()
            && !crate::y_bounds::identity_bounds(bounds)
        {
            let y_z = crate::y_bounds::warp_y(*y, bounds)?;
            self.update_y(&y_z.view());
            return Ok(self);
        }
        self.update_y(y);
        Ok(self)
    }

    pub fn y_std(&self) -> Array1<f64> {
        if self.y_count == 0 {
            return Array1::ones(self.num_metrics.max(1));
        }
        let n = self.y_count as f64;
        let mut std = Array1::zeros(self.num_metrics);
        for m in 0..self.num_metrics {
            let mean = self.y_sum[m] / n;
            let var = (self.y_sumsq[m] / n - mean * mean).max(0.0);
            std[m] = var.sqrt();
        }
        std.mapv(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 })
    }

    #[cfg(test)]
    pub(crate) fn build_candidates(
        &self,
        num_random: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let scales = baux::semantics::neighbor::draw_scales(
            num_random,
            1.0,
            self.infer_aleatoric_variance,
            rng,
        );
        self.scales_params(&scales, 1.0)
    }

    fn scales_params(
        &self,
        scales: &[(f64, f64)],
        d_scale: f64,
    ) -> Result<Vec<ENNParams>, ENNError> {
        if scales.is_empty() {
            return Ok(vec![]);
        }
        let d_scale = if d_scale.is_finite() && d_scale > 0.0 {
            d_scale.max(1.0e-12)
        } else {
            1.0
        };
        let paramss: Vec<ENNParams> = scales
            .iter()
            .filter_map(|&(e, a)| ENNParams::new(self.k, e, a).ok())
            .collect();
        if paramss.is_empty() {
            return ENNParams::new(self.k, 1.0 / d_scale, 0.0)
                .map(|p| vec![p])
                .map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                });
        }
        Ok(paramss)
    }

    fn with_warm(
        &self,
        mut paramss: Vec<ENNParams>,
        params_warm_start: Option<&ENNParams>,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let warm = params_warm_start.or(self.params.as_ref());
        if let Some(warm) = warm {
            paramss.push(
                ENNParams::new(
                    self.k,
                    warm.epistemic_scale,
                    if self.infer_aleatoric_variance {
                        warm.aleatoric_scale
                    } else {
                        0.0
                    },
                )
                .map_err(|error| {
                    ENNError::InvalidParameter(format!("Invalid warm-start params: {error}"))
                })?,
            );
        }
        Ok(paramss)
    }

    fn candidates_key(
        &self,
        num_candidates: usize,
        d_scale: f64,
        params_warm_start: Option<&ENNParams>,
        key: Key,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let eff_scale = if d_scale.is_finite() && d_scale > 0.0 {
            d_scale.max(1.0e-12)
        } else {
            1.0
        };
        let scales = if num_candidates == 0 {
            Vec::new()
        } else {
            draw_key(
                num_candidates,
                eff_scale,
                self.infer_aleatoric_variance,
                key,
            )
        };
        let paramss = self.scales_params(&scales, eff_scale)?;
        self.with_warm(paramss, params_warm_start)
    }

    fn fit_candidates(
        mut self,
        model: &ENN,
        paramss: &[ENNParams],
        indices: &[usize],
    ) -> Result<(Self, ENNParams), ENNError> {
        if self.y_count != model.num_obs() || self.num_metrics != model.num_metrics() {
            self.y_count = model.num_obs();
            self.num_metrics = model.num_metrics();
            self.y_sum = model.y_sum.clone();
            self.y_sumsq = model.y_sumsq.clone();
        }
        let y_std = self.y_std();
        let logliks = crate::fit::rows_loglik(
            model,
            indices,
            paramss,
            indices.len().max(1),
            Key::default(),
            Some(&y_std.view()),
        )?;
        let best_idx = logliks
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .ok_or_else(|| ENNError::InvalidParameter("no valid likelihood candidates".into()))?;
        let best = paramss[best_idx];
        self.params = Some(best);
        Ok((self, best))
    }

    /// Pure functional hyperparameter fit over `model` driven by an immutable
    /// `Key`.
    pub fn ask_key(
        mut self,
        model: &ENN,
        num_candidates: usize,
        num_samples: usize,
        prior: Option<&ENNParams>,
        key: Key,
    ) -> Result<(Self, ENNParams), ENNError> {
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|e| {
                ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
            })?;
            self.params = Some(best);
            return Ok((self, best));
        }
        let (cand_key, sample_key) = key.split();
        let paramss = self.candidates_key(num_candidates, 1.0, prior, cand_key)?;
        let indices = crate::fit::sampled_key(model.len(), num_samples, sample_key);
        self.fit_candidates(model, &paramss, &indices)
    }

    /// Affine-key hyperparameter fit advancing the single-use `AffineKey`.
    pub fn ask_affine(
        self,
        model: &ENN,
        num_candidates: usize,
        num_samples: usize,
        prior: Option<&ENNParams>,
        key: AffineKey,
    ) -> Result<(Self, ENNParams, AffineKey), ENNError> {
        let (draw, next) = key.step();
        let (this, best) = self.ask_key(model, num_candidates, num_samples, prior, draw)?;
        Ok((this, best, next))
    }

    /// Backwards-compatible `&mut dyn RngCore` adapter delegating to pure
    /// [`Self::fit_candidates`].
    pub fn ask(
        mut self,
        model: &ENN,
        num_candidates: usize,
        num_samples: usize,
        prior: Option<&ENNParams>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, ENNParams), ENNError> {
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|e| {
                ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
            })?;
            self.params = Some(best);
            return Ok((self, best));
        }
        let scales = if num_candidates == 0 {
            Vec::new()
        } else {
            baux::semantics::neighbor::draw_scales(
                num_candidates,
                1.0,
                self.infer_aleatoric_variance,
                rng,
            )
        };
        let paramss = self.with_warm(self.scales_params(&scales, 1.0)?, prior)?;
        let indices = baux::semantics::neighbor::sampled_rows(model.len(), num_samples, rng);
        self.fit_candidates(model, &paramss, &indices)
    }

    fn fit_distances(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        paramss: &[ENNParams],
        selected: &[usize],
    ) -> Result<(Self, ENNParams), ENNError> {
        let y_std = self.y_std();
        let logliks =
            crate::fit::dist_rows(distances, y, yvar, paramss, selected, Some(&y_std.view()))?;
        let best_idx = logliks
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .ok_or_else(|| ENNError::InvalidParameter("no valid likelihood candidates".into()))?;
        let best = paramss[best_idx];
        self.params = Some(best);
        Ok((self, best))
    }

    fn fit_k(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        scale_fit: ENNParams,
        max_k: i32,
        selected: &[usize],
    ) -> Result<(Self, ENNParams), ENNError> {
        let largest = usize::try_from(max_k)
            .unwrap_or(usize::MAX)
            .min(y.nrows() - 1);
        let candidates = (1..=largest)
            .map(|k| {
                ENNParams::new(
                    k as i32,
                    scale_fit.epistemic_scale,
                    scale_fit.aleatoric_scale,
                )
                .map_err(|error| ENNError::InvalidParameter(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let y_std = self.y_std();
        let scores = crate::fit::dist_rows(
            distances,
            y,
            yvar,
            &candidates,
            selected,
            Some(&y_std.view()),
        )?;
        let best = candidates
            .into_iter()
            .zip(scores)
            .filter(|(_, score)| score.is_finite())
            .max_by(|(left, left_score), (right, right_score)| {
                left_score
                    .total_cmp(right_score)
                    .then_with(|| right.k_neighbors.cmp(&left.k_neighbors))
            })
            .map(|(params, _)| params)
            .ok_or_else(|| ENNError::InvalidParameter("no valid neighbor count".into()))?;
        self.params = Some(best);
        Ok((self, best))
    }

    /// Pure functional fit from outcomes and exact pairwise squared distances.
    pub fn distances_key(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        key: Key,
    ) -> Result<(Self, ENNParams), ENNError> {
        if y.nrows() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|error| {
                ENNError::InvalidParameter(format!("Failed to create default params: {error}"))
            })?;
            self.params = Some(best);
            return Ok((self, best));
        }
        if num_samples == 0 {
            return Err(ENNError::InvalidParameter(
                "nonempty parameters and p > 0 required".into(),
            ));
        }
        self.y_stats(y);
        let d_scale = characteristic_distance(distances);
        let (cand_key, loo_key) = key.split();
        let paramss = self.candidates_key(num_candidates, d_scale, params_warm_start, cand_key)?;
        let selected = crate::fit::sampled_key(y.nrows(), num_samples, loo_key);
        self.fit_distances(distances, y, yvar, &paramss, &selected)
    }

    /// Fit ENN scales from outcomes and their exact pairwise squared distances.
    pub fn ask_distances(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, ENNParams), ENNError> {
        if y.nrows() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|error| {
                ENNError::InvalidParameter(format!("Failed to create default params: {error}"))
            })?;
            self.params = Some(best);
            return Ok((self, best));
        }
        if num_samples == 0 {
            return Err(ENNError::InvalidParameter(
                "nonempty parameters and p > 0 required".into(),
            ));
        }
        self.y_stats(y);
        let d_scale = characteristic_distance(distances);
        let eff_scale = if d_scale.is_finite() && d_scale > 0.0 {
            d_scale.max(1.0e-12)
        } else {
            1.0
        };
        let scales = if num_candidates == 0 {
            Vec::new()
        } else {
            baux::semantics::neighbor::draw_scales(
                num_candidates,
                eff_scale,
                self.infer_aleatoric_variance,
                rng,
            )
        };
        let paramss = self.with_warm(self.scales_params(&scales, eff_scale)?, params_warm_start)?;
        let selected = baux::semantics::neighbor::sampled_rows(y.nrows(), num_samples, rng);
        self.fit_distances(distances, y, yvar, &paramss, &selected)
    }

    /// Pure functional adaptive `(scales, k)` leave-one-out fit driven by
    /// `Key`.
    pub fn adaptive_key(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        max_k: i32,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        key: Key,
    ) -> Result<(Self, ENNParams), ENNError> {
        if max_k <= 0 {
            return Err(ENNError::InvalidParameter(
                "adaptive neighbor fitting requires max_k > 0".into(),
            ));
        }
        let (scale_key, k_key) = key.split();
        let prev_k = self.k;
        self.k = max_k;
        let (mut this, mut scale_fit) = self.distances_key(
            distances,
            y,
            yvar,
            num_candidates,
            num_samples,
            params_warm_start,
            scale_key,
        )?;
        this.k = prev_k;
        if y.nrows() <= 2 {
            scale_fit.k_neighbors = scale_fit
                .k_neighbors
                .min((y.nrows().saturating_sub(1)).max(1) as i32);
            this.params = Some(scale_fit);
            return Ok((this, scale_fit));
        }
        let selected = crate::fit::sampled_key(y.nrows(), num_samples, k_key);
        this.fit_k(distances, y, yvar, scale_fit, max_k, &selected)
    }

    /// Fit variance scales and then select `k` by the same leave-one-out
    /// predictive likelihood. The configured value is an upper bound; every
    /// feasible integer count participates rather than a hand-written grid.
    pub fn ask_adaptive(
        mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        max_k: i32,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, ENNParams), ENNError> {
        if max_k <= 0 {
            return Err(ENNError::InvalidParameter(
                "adaptive neighbor fitting requires max_k > 0".into(),
            ));
        }
        let prev_k = self.k;
        self.k = max_k;
        let (mut this, mut scale_fit) = self.ask_distances(
            distances,
            y,
            yvar,
            num_candidates,
            num_samples,
            params_warm_start,
            rng,
        )?;
        this.k = prev_k;
        if y.nrows() <= 2 {
            scale_fit.k_neighbors = scale_fit
                .k_neighbors
                .min((y.nrows().saturating_sub(1)).max(1) as i32);
            this.params = Some(scale_fit);
            return Ok((this, scale_fit));
        }
        let selected = baux::semantics::neighbor::sampled_rows(y.nrows(), num_samples, rng);
        this.fit_k(distances, y, yvar, scale_fit, max_k, &selected)
    }
}

fn draw_key(
    num_candidates: usize,
    d_scale: f64,
    infer_aleatoric: bool,
    key: Key,
) -> Vec<(f64, f64)> {
    (0..num_candidates)
        .map(|i| {
            baux::semantics::optimizer::surrogate::scale_prior(key, i, d_scale, infer_aleatoric)
        })
        .collect()
}

fn characteristic_distance(distances: &ArrayView2<f64>) -> f64 {
    baux::semantics::neighbor::characteristic_distance(distances.nrows(), |row, col| {
        distances[[row, col]]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::test_helpers::test_model as create_test_model;
    use ndarray::{Array2, array, s};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_ennfitteraskbasic() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        let fitter = fitter.observe_y(&ty.view());

        let (_fitter, result) = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
        assert!(result.aleatoric_scale >= 0.0);
    }

    #[test]
    fn test_005() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        let fitter = fitter.observe_y(&ty.view());

        let warm_start = ENNParams::new(2, 1.5, 0.2).unwrap();

        let (_fitter, result) = fitter
            .ask(&model, 5, 3, Some(&warm_start), &mut rng)
            .unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn test_006() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        let fitter = fitter.observe_y(&ty.view());

        let (_fitter, result) = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
        assert_eq!(result.aleatoric_scale, 0.0);
    }

    #[test]
    fn test_007() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let train_y = array![[0.0, 1.0], [1.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 1.5]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        let fitter = fitter.observe_y(&ty.view());

        let (_fitter, result) = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn y_finite() {
        let fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0]];
        let y = array![[f64::NAN]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn x_finite() {
        let fitter = ENNFitter::new(2, true);
        let x = array![[f64::NAN, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn tell_yvar() {
        let fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.clone().tell(&x.view(), &y.view(), None).is_err());
        let yvar = array![[0.1, 0.2]];
        assert!(
            fitter
                .clone()
                .tell(&x.view(), &y.view(), Some(&yvar.view()))
                .is_err()
        );
        let yvar_bad = array![[f64::INFINITY]];
        assert!(
            fitter
                .tell(&x.view(), &y.view(), Some(&yvar_bad.view()))
                .is_err()
        );
    }

    #[test]
    fn ask_start() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = ENN::new(
            train_x.clone(),
            train_y.clone(),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let fitter = ENNFitter::new(2, true)
            .tell(&train_x.view(), &train_y.view(), None)
            .unwrap();
        let warm = ENNParams::new(2, 2.5, 0.3).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let (_fitter, p) = fitter.ask(&model, 0, 2, Some(&warm), &mut rng).unwrap();
        assert_eq!(p.k_neighbors, 2);
        assert!((p.epistemic_scale - 2.5).abs() < 1e-12);
    }

    #[test]
    fn ask_inferred() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        let fitter = fitter.observe_y(&ty.view());
        let warm = ENNParams::new(2, 2.5, 9.9).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let (_fitter, p) = fitter.ask(&model, 2, 2, Some(&warm), &mut rng).unwrap();
        assert_eq!(p.aleatoric_scale, 0.0);
    }

    #[test]
    fn ask_2() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let fitter = ENNFitter::new(3, true);
        let mut rng = StdRng::seed_from_u64(1);
        let (_fitter, p) = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();
        assert_eq!(p.k_neighbors, 3);
        assert!((p.epistemic_scale - 1.0).abs() < 1e-12);
        assert!((p.aleatoric_scale - 0.0).abs() < 1e-12);
    }

    #[test]
    fn y_std() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0], [1.5]];
        let _model = ENN::new(
            train_x.clone(),
            train_y.clone(),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let mut fitter = ENNFitter::new(2, true);
        for (i, row) in train_y.axis_iter(Axis(0)).enumerate() {
            let y_row = row.insert_axis(Axis(0));
            let x_row = train_x.slice(s![i..i + 1, ..]);
            fitter = fitter.tell(&x_row, &y_row, None).unwrap();
        }
        let batch_std = train_y.std_axis(Axis(0), 0.0);
        let inc_std = fitter.y_std();
        for (a, b) in inc_std.iter().zip(batch_std.iter()) {
            if *b > 1e-10 {
                assert!(
                    (a - b).abs() < 1e-10,
                    "incremental std {a} vs batch std {b}"
                );
            } else {
                assert!(
                    (*a - 1.0).abs() < 1e-10,
                    "zero-variance metric should clamp to 1.0"
                );
            }
        }
    }

    #[test]
    fn fitter_updates() {
        let fitter = ENNFitter::new(2, true);
        let y = array![[1.0], [2.0]];
        let fitter = fitter.observe_y(&y.view());
        let mut rng = StdRng::seed_from_u64(0);
        let cands = fitter.build_candidates(3, &mut rng).unwrap();
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn adaptive_fit() {
        let x = [0.0f64, 0.2, 0.6, 1.0];
        let distances = Array2::from_shape_fn((x.len(), x.len()), |(i, j)| (x[i] - x[j]).powi(2));
        let y = Array2::from_shape_fn((x.len(), 1), |(i, _)| x[i]);
        let variance = Array2::from_elem((x.len(), 1), 1.0e-4);
        let fitter = ENNFitter::new(3, true);
        let mut rng = StdRng::seed_from_u64(29);
        let (fitter, fitted) = fitter
            .ask_adaptive(
                &distances.view(),
                &y.view(),
                Some(&variance.view()),
                3,
                16,
                4,
                None,
                &mut rng,
            )
            .unwrap();
        assert!((1..=3).contains(&fitted.k_neighbors));
        assert_eq!(fitter.params(), Some(&fitted));
    }

    #[test]
    fn pure_fit() {
        let model = create_test_model();
        let key = crate::hash::Key::new(42);
        let p1 = fit_model(&model, 2, 5, 3, true, None, key).unwrap();
        let p2 = fit_model(&model, 2, 5, 3, true, None, key).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.k_neighbors, 2);

        let fitter = ENNFitter::new(2, true);
        let (fitter_clone, p3) = fitter.clone().ask_key(&model, 5, 3, None, key).unwrap();
        assert_eq!(p1, p3);
        assert_eq!(fitter_clone.params(), Some(&p1));

        let (_fitter_aff, p4, _next) = fitter
            .ask_affine(&model, 5, 3, None, AffineKey::from_key(key))
            .unwrap();
        assert_eq!(p4.k_neighbors, 2);
    }
}
