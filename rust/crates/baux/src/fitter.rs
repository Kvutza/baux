//! Stateful ENN hyperparameter fitting with incremental statistics.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::Rng;

use crate::error::ENNError;
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
    key: crate::hash::Key,
) -> Result<ENNParams, ENNError> {
    let mut rng = key.prng();
    let mut fitter = ENNFitter::new(k, infer_aleatoric);
    if model.bounded_outputs() {
        fitter = fitter.with_bounds(Some(model.y_bounds().clone()));
    }
    fitter.ask(model, num_candidates, num_samples, warm.as_ref(), &mut rng)
}

/// Stateful ENN fitter: running `y` moments and warm-start params.
#[derive(Debug, Clone)]
pub struct ENNFitter {
    k: i32,
    infer_aleatoric_variance: bool,
    params: Option<ENNParams>,
    y_bounds: Option<Array2<f64>>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
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
            y_sum: Array1::zeros(0),
            y_sumsq: Array1::zeros(0),
            y_count: 0,
            num_metrics: 0,
        }
    }

    pub fn with_bounds(mut self, bounds: Option<Array2<f64>>) -> Self {
        self.y_bounds = bounds;
        self
    }

    pub fn set_bounds(&mut self, bounds: Option<Array2<f64>>) {
        self.y_bounds = bounds;
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
            y_bounds,
            y_sum,
            y_sumsq,
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
        self.y_sum = y.sum_axis(Axis(0));
        self.y_sumsq = y.mapv(|v| v * v).sum_axis(Axis(0));
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
        for row in y_new.axis_iter(Axis(0)) {
            for m in 0..self.num_metrics {
                let v = row[m];
                self.y_sum[m] += v;
                self.y_sumsq[m] += v * v;
            }
            self.y_count += 1;
        }
    }

    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
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
        if let Some(bounds) = self.y_bounds.as_ref() {
            if !crate::y_bounds::identity_bounds(bounds) {
                let y_z = crate::y_bounds::warp_y(*y, bounds)?;
                self.update_y(&y_z.view());
                return Ok(());
            }
        }
        self.update_y(y);
        Ok(())
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
    pub(crate) fn build_candidates<R: Rng>(
        &self,
        num_random: usize,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        self.scaled_candidates(num_random, 1.0, rng)
    }

    fn scaled_candidates<R: Rng>(
        &self,
        num_random: usize,
        d_scale: f64,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        if num_random == 0 {
            return Ok(vec![]);
        }
        let d_scale = if d_scale.is_finite() && d_scale > 0.0 {
            d_scale.max(1.0e-12)
        } else {
            1.0
        };
        let paramss: Vec<ENNParams> = baux::semantics::neighbor::draw_scales(
            num_random,
            d_scale,
            self.infer_aleatoric_variance,
            rng,
        )
        .into_iter()
        .filter_map(|(e, a)| ENNParams::new(self.k, e, a).ok())
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

    fn candidates<R: Rng>(
        &self,
        num_candidates: usize,
        d_scale: f64,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let mut paramss = self.scaled_candidates(num_candidates, d_scale, rng)?;
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

    pub fn ask<R: Rng>(
        &mut self,
        model: &ENN,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<ENNParams, ENNError> {
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|e| {
                ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
            })?;
            self.params = Some(best);
            return Ok(best);
        }
        if self.y_count != model.num_obs() || self.num_metrics != model.num_metrics() {
            self.y_count = model.num_obs();
            self.num_metrics = model.num_metrics();
            self.y_sum = model.y_sum.clone();
            self.y_sumsq = model.y_sumsq.clone();
        }
        let paramss = self.candidates(num_candidates, 1.0, params_warm_start, rng)?;
        let indices = baux::semantics::neighbor::sampled_rows(model.len(), num_samples, rng);
        let y_std = self.y_std();
        let logliks = crate::fit::row_loglik(
            model,
            &indices,
            &paramss,
            num_samples,
            rng,
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
        Ok(best)
    }

    /// Fit ENN scales from outcomes and their exact pairwise squared distances.
    pub fn ask_distances<R: Rng>(
        &mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<ENNParams, ENNError> {
        if y.nrows() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|error| {
                ENNError::InvalidParameter(format!("Failed to create default params: {error}"))
            })?;
            self.params = Some(best);
            return Ok(best);
        }
        self.y_stats(y);
        let d_scale = characteristic_distance(distances);
        let paramss = self.candidates(num_candidates, d_scale, params_warm_start, rng)?;
        let y_std = self.y_std();
        let logliks = crate::fit::distance_loglik(
            distances,
            y,
            yvar,
            &paramss,
            num_samples,
            rng,
            Some(&y_std.view()),
        )?;
        let best_idx = logliks
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .ok_or_else(|| ENNError::InvalidParameter("no valid likelihood candidates".into()))?;
        let best = paramss[best_idx];
        self.params = Some(best);
        Ok(best)
    }

    /// Fit variance scales and then select `k` by the same leave-one-out
    /// predictive likelihood. The configured value is an upper bound; every
    /// feasible integer count participates rather than a hand-written grid.
    pub fn ask_adaptive<R: Rng>(
        &mut self,
        distances: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        max_k: i32,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<ENNParams, ENNError> {
        if max_k <= 0 {
            return Err(ENNError::InvalidParameter(
                "adaptive neighbor fitting requires max_k > 0".into(),
            ));
        }
        let prev_k = self.k;
        self.k = max_k;
        let scale_fit = self.ask_distances(
            distances,
            y,
            yvar,
            num_candidates,
            num_samples,
            params_warm_start,
            rng,
        );
        self.k = prev_k;
        let mut scale_fit = scale_fit?;
        if y.nrows() <= 2 {
            scale_fit.k_neighbors = scale_fit
                .k_neighbors
                .min((y.nrows().saturating_sub(1)).max(1) as i32);
            self.params = Some(scale_fit);
            return Ok(scale_fit);
        }
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
        let scores = crate::fit::distance_loglik(
            distances,
            y,
            yvar,
            &candidates,
            num_samples,
            rng,
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
        Ok(best)
    }
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
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
        assert!(result.aleatoric_scale >= 0.0);
    }

    #[test]
    fn test_005() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let warm_start = ENNParams::new(2, 1.5, 0.2).unwrap();

        let result = fitter
            .ask(&model, 5, 3, Some(&warm_start), &mut rng)
            .unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn test_006() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

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
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn y_finite() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0]];
        let y = array![[f64::NAN]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn x_finite() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[f64::NAN, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn tell_yvar() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
        let yvar = array![[0.1, 0.2]];
        assert!(
            fitter
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
        let mut fitter = ENNFitter::new(2, true);
        fitter.tell(&train_x.view(), &train_y.view(), None).unwrap();
        let warm = ENNParams::new(2, 2.5, 0.3).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let p = fitter.ask(&model, 0, 2, Some(&warm), &mut rng).unwrap();
        assert_eq!(p.k_neighbors, 2);
        assert!((p.epistemic_scale - 2.5).abs() < 1e-12);
    }

    #[test]
    fn ask_inferred() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let mut fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());
        let warm = ENNParams::new(2, 2.5, 9.9).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let p = fitter.ask(&model, 2, 2, Some(&warm), &mut rng).unwrap();
        assert_eq!(p.aleatoric_scale, 0.0);
    }

    #[test]
    fn ask_2() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let mut fitter = ENNFitter::new(3, true);
        let mut rng = StdRng::seed_from_u64(1);
        let p = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();
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
            fitter.tell(&x_row, &y_row, None).unwrap();
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
        let mut fitter = ENNFitter::new(2, true);
        let y = array![[1.0], [2.0]];
        fitter.update_y(&y.view());
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
        let mut fitter = ENNFitter::new(3, true);
        let mut rng = StdRng::seed_from_u64(29);
        let fitted = fitter
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

        let mut fitter = ENNFitter::new(2, true);
        let mut rng = key.prng();
        let p3 = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();
        assert_eq!(p1, p3);
    }
}
