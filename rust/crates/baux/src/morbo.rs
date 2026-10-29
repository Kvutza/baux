//! Morbo trust region: Chebyshev scalarization over an inner TuRBO trust
//! region.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;
use rand::RngCore;

#[cfg(test)]
use crate::trust_region::TRLengthConfig;
pub use crate::trust_region::{MorboTRSettings, Rescalarize};
use crate::trust_region::{TrustRegionError, TurboTrustRegion};

#[derive(Debug, Clone)]
pub struct MorboTrustRegion {
    inner: TurboTrustRegion,
    num_metrics: usize,
    alpha: f64,
    rescalarize: Rescalarize,
    #[allow(dead_code)]
    noise_aware: bool,
    weights: Array1<f64>,
    y_min: Option<Array1<f64>>,
    y_max: Option<Array1<f64>>,
    incumbent_y_raw: Option<Array1<f64>>,
    score_min: Option<f64>,
    score_max: Option<f64>,
    score_count: usize,
}

impl MorboTrustRegion {
    pub fn new(
        num_dim: usize,
        settings: MorboTRSettings,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        if settings.num_metrics < 2 {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_metrics must be >= 2 for MORBO, got {}",
                settings.num_metrics
            )));
        }
        let inner = TurboTrustRegion::new(num_dim, settings.length);
        let mut morbo = Self {
            inner,
            num_metrics: settings.num_metrics,
            alpha: settings.alpha,
            rescalarize: settings.rescalarize,
            noise_aware: settings.noise_aware,
            weights: Array1::zeros(settings.num_metrics),
            y_min: None,
            y_max: None,
            incumbent_y_raw: None,
            score_min: None,
            score_max: None,
            score_count: 0,
        };
        morbo.resample_weights(rng);
        Ok(morbo)
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn length(&self) -> f64 {
        self.inner.length()
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: &[f64]) -> Result<(), TrustRegionError> {
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
        self.weights = Array1::from_iter(weights.iter().map(|weight| weight / sum));
        Ok(())
    }

    pub fn y_min(&self) -> Option<&Array1<f64>> {
        self.y_min.as_ref()
    }

    pub fn y_max(&self) -> Option<&Array1<f64>> {
        self.y_max.as_ref()
    }

    pub fn rescalarize(&self) -> Rescalarize {
        self.rescalarize
    }

    pub fn resample_weights(&mut self, rng: &mut dyn RngCore) {
        self.weights = sample_weights(rng, self.num_metrics);
    }

    pub fn set_arms(&mut self, num_arms: usize) {
        self.inner.set_arms(num_arms);
    }

    pub fn set_tolerance(&mut self, failures: usize) -> Result<(), TrustRegionError> {
        self.inner.set_tolerance(failures)
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

    pub fn restart(&mut self, rng: Option<&mut dyn RngCore>) {
        self.y_min = None;
        self.y_max = None;
        self.incumbent_y_raw = None;
        self.score_min = None;
        self.score_max = None;
        self.score_count = 0;
        self.inner.restart();
        if let Some(rng) = rng {
            if self.rescalarize == Rescalarize::OnRestart {
                self.resample_weights(rng);
            }
        }
    }

    pub fn compute_bounds(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        self.inner.compute_bounds(x_center, lengthscales)
    }

    pub fn scalarize(
        &self,
        y: &ArrayView2<f64>,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        let (y_min, y_max) = self
            .y_min
            .as_ref()
            .zip(self.y_max.as_ref())
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

    /// Chebyshev scores using the minimum and maximum of `y` itself.
    pub fn scalarize_local(&self, y: &ArrayView2<f64>) -> Result<Array1<f64>, TrustRegionError> {
        let m = y.ncols();
        let mut y_min = Array1::from_elem(m, f64::INFINITY);
        let mut y_max = Array1::from_elem(m, f64::NEG_INFINITY);
        for i in 0..y.nrows() {
            for j in 0..m {
                y_min[j] = y_min[j].min(y[[i, j]]);
                y_max[j] = y_max[j].max(y[[i, j]]);
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

    pub fn update_incremental(&mut self, y_new: &ArrayView2<f64>) {
        if y_new.nrows() == 0 {
            return;
        }
        if self.y_min.is_none() || self.y_max.is_none() {
            let mut y_min = Array1::from_elem(self.num_metrics, f64::INFINITY);
            let mut y_max = Array1::from_elem(self.num_metrics, f64::NEG_INFINITY);
            for row in 0..y_new.nrows() {
                for m in 0..self.num_metrics {
                    let v = y_new[[row, m]];
                    if v < y_min[m] {
                        y_min[m] = v;
                    }
                    if v > y_max[m] {
                        y_max[m] = v;
                    }
                }
            }
            self.y_min = Some(y_min);
            self.y_max = Some(y_max);
            return;
        }
        let y_min = self.y_min.as_mut().expect("y_min");
        let y_max = self.y_max.as_mut().expect("y_max");
        for row in 0..y_new.nrows() {
            for m in 0..self.num_metrics {
                let v = y_new[[row, m]];
                if v < y_min[m] {
                    y_min[m] = v;
                }
                if v > y_max[m] {
                    y_max[m] = v;
                }
            }
        }
    }

    pub fn update_only(
        &mut self,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), TrustRegionError> {
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
            self.inner.restart();
            return Ok(());
        }
        let (y_min, y_max) = self
            .y_min
            .as_ref()
            .zip(self.y_max.as_ref())
            .ok_or_else(|| {
                TrustRegionError::InvalidState("update_only before ranges initialized".to_string())
            })?;
        if self.incumbent_y_raw.is_none() {
            self.incumbent_y_raw = Some(y_incumbent.to_owned());
            let score = scalarize_ranges(
                &y_incumbent.view().insert_axis(ndarray::Axis(0)),
                &y_min.view(),
                &y_max.view(),
                &self.weights.view(),
                self.alpha,
                self.num_metrics,
                true,
            )?[0];
            push_score(self, num_obs, score)
                .map_err(|e| TrustRegionError::InvalidState(e.to_string()))?;
            return Ok(());
        }
        let incumbent = self.incumbent_y_raw.as_ref().expect("incumbent");
        let stacked = ndarray::stack(ndarray::Axis(0), &[incumbent.view(), y_incumbent.view()])
            .map_err(|e| TrustRegionError::InvalidState(e.to_string()))?;
        let scores = scalarize_ranges(
            &stacked.view(),
            &y_min.view(),
            &y_max.view(),
            &self.weights.view(),
            self.alpha,
            self.num_metrics,
            true,
        )?;
        let old_score = scores[0];
        let new_score = scores[1];
        self.inner.set_best(old_score);
        push_score(self, num_obs, new_score)?;
        if new_score > old_score {
            self.incumbent_y_raw = Some(y_incumbent.to_owned());
        }
        Ok(())
    }

    pub fn rescalarize_weights(&mut self, num_obs: usize) -> Result<(), TrustRegionError> {
        let Some(ref y_inc) = self.incumbent_y_raw else {
            return Ok(());
        };
        let (Some(y_min), Some(y_max)) = (self.y_min.as_ref(), self.y_max.as_ref()) else {
            return Ok(());
        };
        let score = scalarize_ranges(
            &y_inc.view().insert_axis(ndarray::Axis(0)),
            &y_min.view(),
            &y_max.view(),
            &self.weights.view(),
            self.alpha,
            self.num_metrics,
            true,
        )?[0];
        self.inner.set_best(score);
        push_score(self, num_obs, score)
    }

    pub fn update(
        &mut self,
        y_obs: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
    ) -> Result<(), TrustRegionError> {
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
            self.inner.restart();
            return Ok(());
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
                if v < y_min[m] {
                    y_min[m] = v;
                }
                if v > y_max[m] {
                    y_max[m] = v;
                }
            }
        }
        self.y_min = Some(y_min);
        self.y_max = Some(y_max);

        if prev_n == 0 || self.incumbent_y_raw.is_none() {
            self.incumbent_y_raw = Some(y_incumbent.to_owned());
            let score = scalarize_ranges(
                &y_incumbent.view().insert_axis(ndarray::Axis(0)),
                &self.y_min.as_ref().unwrap().view(),
                &self.y_max.as_ref().unwrap().view(),
                &self.weights.view(),
                self.alpha,
                self.num_metrics,
                true,
            )?[0];
            push_score(self, n, score)
                .map_err(|e| TrustRegionError::InvalidState(e.to_string()))?;
            return Ok(());
        }

        let incumbent = self.incumbent_y_raw.as_ref().unwrap();
        let stacked = ndarray::stack(ndarray::Axis(0), &[incumbent.view(), y_incumbent.view()])
            .map_err(|e| TrustRegionError::InvalidState(e.to_string()))?;
        let scores = scalarize_ranges(
            &stacked.view(),
            &self.y_min.as_ref().unwrap().view(),
            &self.y_max.as_ref().unwrap().view(),
            &self.weights.view(),
            self.alpha,
            self.num_metrics,
            true,
        )?;
        let old_score = scores[0];
        let new_score = scores[1];
        self.inner.set_best(old_score);
        push_score(self, n, new_score)?;
        if new_score > old_score {
            self.incumbent_y_raw = Some(y_incumbent.to_owned());
        }
        Ok(())
    }
}

fn push_score(
    morbo: &mut MorboTrustRegion,
    num_obs: usize,
    score: f64,
) -> Result<(), TrustRegionError> {
    let scale = if morbo.score_count >= 2 {
        match (morbo.score_min, morbo.score_max) {
            (Some(low), Some(high)) => (high - low).max(1e-6),
            _ => 0.0,
        }
    } else {
        0.0
    };
    morbo.inner.update_scalar(num_obs, score, scale)?;
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
    Ok(())
}

fn sample_weights(rng: &mut dyn RngCore, n: usize) -> Array1<f64> {
    let mut samples = Vec::with_capacity(n);
    let mut sum = 0.0;
    for _ in 0..n {
        let u: f64 = rng.r#gen();
        let g = (-u.ln()).max(1e-300);
        samples.push(g);
        sum += g;
    }
    Array1::from_vec(samples.iter().map(|&v| v / sum).collect())
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
    if y.ncols() != num_metrics || y_min.len() != num_metrics || y_max.len() != num_metrics {
        return Err(TrustRegionError::InvalidParameter(format!(
            "shape mismatch: y {:?}, y_min {}, y_max {}",
            y.shape(),
            y_min.len(),
            y_max.len()
        )));
    }
    let n = y.nrows();
    let mut scores = Array1::zeros(n);
    for row in 0..n {
        let mut t_min = f64::INFINITY;
        let mut t_sum = 0.0;
        for m in 0..num_metrics {
            let denom = y_max[m] - y_min[m];
            let z = if denom <= 0.0 {
                0.5
            } else {
                let mut z = (y[[row, m]] - y_min[m]) / denom;
                if clip {
                    z = z.clamp(0.0, 1.0);
                }
                z
            };
            let t = z * weights[m];
            t_min = t_min.min(t);
            t_sum += t;
        }
        scores[row] = t_min + alpha * t_sum;
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
        let mut rng = StdRng::seed_from_u64(7);
        let mut tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
        tr.set_weights(&[0.5, 0.5]).unwrap();
        tr.update_incremental(&array![[0.0, 0.0], [100.0, 10.0]].view());

        let candidates = array![[10.0, 9.0], [9.0, 10.0]];
        let local = tr.scalarize_local(&candidates.view()).unwrap();
        let global = tr.scalarize(&candidates.view(), true).unwrap();

        assert!((local[0] - 0.05).abs() < 1e-12);
        assert!((local[1] - 0.05).abs() < 1e-12);
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
        let mut rng = StdRng::seed_from_u64(7);
        let result = MorboTrustRegion::new(2, settings, &mut rng);
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
        let mut rng = StdRng::seed_from_u64(99);
        let result = MorboTrustRegion::new(2, settings, &mut rng);
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
        let mut rng = StdRng::seed_from_u64(1);
        let mut tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
        tr.set_arms(2);
        let y = array![[1.0, 2.0], [2.0, 3.0]];
        let inc = array![2.0, 3.0];
        tr.update(&y.view(), &inc.view()).unwrap();
        assert!(tr.incumbent_y_raw.is_some());
        assert_eq!(tr.score_count, 1);
        assert_eq!(tr.score_min, tr.score_max);

        tr.restart(None);
        assert_eq!(tr.score_count, 0);
        assert_eq!(tr.score_min, None);
        assert_eq!(tr.score_max, None);
    }
}
