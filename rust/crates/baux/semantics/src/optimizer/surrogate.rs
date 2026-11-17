//! Bounded neighbor-value surrogate estimation used by the optimizer.

mod loo_fit;

use baux::semantics::abi::{ACQ_PARETO, ACQ_RANDOM, ACQ_THOMPSON, ACQ_UCB};
use baux::semantics::key::Key;
pub use baux::semantics::neighbor::{
    Dissimilarity, DistanceMetric, MAX_HISTORY, MetricKind, Posterior, TopK, UncertaintyLaw,
};
use baux::semantics::real::Real;
use core::cmp::Ordering;
use deser::{Deserialize, Serialize};
pub use loo_fit::{loo_point, loo_slice, scale_prior};

pub const INITIAL_OBSERVATIONS: usize = 32;
pub const DEFAULT_BETA: f32 = 1.0;

/// NaN-safe maximization score wrapper parameterized over [`Real`]: all `NaN`
/// bit patterns compare equal to each other and strictly below
/// `T::NEG_INFINITY`, while non-`NaN` values follow `T::total_cmp`.
#[derive(Clone, Copy, Debug, Default)]
pub struct BetterScore<T: Real = f64>(pub T);

impl<T: Real> BetterScore<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn value(self) -> T {
        self.0
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.0.is_nan()
    }
}

impl<T: Real> PartialEq for BetterScore<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: Real> Eq for BetterScore<T> {}

impl<T: Real> PartialOrd for BetterScore<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Real> Ord for BetterScore<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.0.total_cmp(&other.0),
        }
    }
}

/// A scalar value prediction with its standard error.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValuePrediction {
    pub mean: f32,
    pub standard_error: f32,
}

/// Resolved surrogate settings transported unchanged to accelerator hosts.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[deser(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SurrogateSpec {
    pub neighbors: usize,
    /// Coefficient multiplying squared parameter distance in the variance
    /// proxy.
    pub epistemic: f32,
    /// Additive observation-noise variance proxy before output scaling.
    pub aleatoric: f32,
    /// Output standard-deviation scale, not a variance.
    pub scale: f32,
    pub fit_candidates: usize,
    pub fit_samples: usize,
    pub fit: bool,
    pub adaptive: bool,
    #[deser(default)]
    pub metric: DistanceMetric,
}

impl Default for SurrogateSpec {
    fn default() -> Self {
        Self {
            neighbors: INITIAL_OBSERVATIONS,
            epistemic: 0.05,
            aleatoric: 0.05,
            scale: 1.0,
            fit_candidates: 30,
            fit_samples: 10,
            fit: false,
            adaptive: false,
            metric: DistanceMetric::Euclidean,
        }
    }
}

impl SurrogateSpec {
    pub fn check(self) -> Result<Self, &'static str> {
        let valid_metric = match self.metric {
            DistanceMetric::Euclidean => true,
            DistanceMetric::Conformal { local_k }
            | DistanceMetric::Hyperbolic { local_k }
            | DistanceMetric::Wasserstein { local_k } => (1..=MAX_HISTORY).contains(&local_k),
        };
        if !(1..=MAX_HISTORY).contains(&self.neighbors)
            || !valid_metric
            || self.fit_candidates == 0
            || self.fit_samples == 0
            || [self.epistemic, self.aleatoric, self.scale]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid resident surrogate settings");
        }
        Ok(self)
    }

    pub fn validate(self) -> Result<Self, String> {
        self.check().map_err(Into::into)
    }
}

/// Acquisition rule used to score candidate rows in a proposal pool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcquisitionRule {
    Ucb { beta: f32 },
    Thompson { seed: u64 },
    Pareto { seed: u64 },
    Random { seed: u64 },
}

impl Default for AcquisitionRule {
    #[inline]
    fn default() -> Self {
        Self::Ucb { beta: DEFAULT_BETA }
    }
}

impl AcquisitionRule {
    #[inline]
    pub const fn code(self) -> u32 {
        match self {
            Self::Ucb { .. } => ACQ_UCB,
            Self::Thompson { .. } => ACQ_THOMPSON,
            Self::Pareto { .. } => ACQ_PARETO,
            Self::Random { .. } => ACQ_RANDOM,
        }
    }

    #[inline]
    pub const fn from_code(code: u32, beta: f32, seed: u64) -> Self {
        match code {
            ACQ_THOMPSON => Self::Thompson { seed },
            ACQ_PARETO => Self::Pareto { seed },
            ACQ_RANDOM => Self::Random { seed },
            _ => Self::Ucb { beta },
        }
    }
}

/// Pure `#![no_std]` UCB acquisition score (`mean + beta * max(se, 0)`).
#[inline]
pub fn score_ucb<T: Real>(mean: T, se: T, beta: T) -> T {
    if !mean.is_finite() || !se.is_finite() || !beta.is_finite() {
        T::NAN
    } else {
        mean + beta * se.max(T::ZERO)
    }
}

/// Pure `#![no_std]` stateless posterior-normal Thompson draw via `Key::fold`.
#[inline]
pub fn score_thompson<T: Real>(key: Key, cand: usize, mean: T, se: T) -> T {
    if !mean.is_finite() || !se.is_finite() {
        T::NAN
    } else {
        mean + se.max(T::ZERO) * key.fold(cand as u64).normal::<T>()
    }
}

/// Strict 2D maximization Pareto dominance on `(mean, se)` with NaN-safety.
#[inline]
pub fn pareto_dom<T: Real>(mean_a: T, se_a: T, mean_b: T, se_b: T) -> bool {
    if mean_a.is_nan() || se_a.is_nan() {
        return false;
    }
    if mean_b.is_nan() || se_b.is_nan() {
        return true;
    }
    mean_a >= mean_b && se_a >= se_b && (mean_a > mean_b || se_a > se_b)
}

fn better_cand<T: Real>(
    score_a: BetterScore<T>,
    tie_a: u64,
    idx_a: usize,
    score_b: BetterScore<T>,
    tie_b: u64,
    idx_b: usize,
) -> bool {
    match score_a.cmp(&score_b) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match tie_a.cmp(&tie_b) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => idx_a < idx_b,
        },
    }
}

/// Allocation-free `#![no_std]` top-`k` (`k = out.len().min(scores.len())`)
/// maximization selector using NaN-safe [`BetterScore`] and `Key::fold`
/// tie-breaking over [`Real`].
pub fn select_topk<T: Real>(scores: &[T], key: Key, out: &mut [usize]) -> usize {
    let k = out.len().min(scores.len());
    if k == 0 {
        return 0;
    }
    let mut len = 0usize;
    for (idx, &raw) in scores.iter().enumerate() {
        let sc = BetterScore(raw);
        let tie = key.fold(idx as u64).seed();
        if len == k {
            let last = out[len - 1];
            let last_sc = BetterScore(scores[last]);
            let last_tie = key.fold(last as u64).seed();
            if !better_cand(sc, tie, idx, last_sc, last_tie, last) {
                continue;
            }
        } else {
            len += 1;
        }
        let mut pos = len - 1;
        while pos > 0 {
            let prev = out[pos - 1];
            let prev_sc = BetterScore(scores[prev]);
            let prev_tie = key.fold(prev as u64).seed();
            if better_cand(sc, tie, idx, prev_sc, prev_tie, prev) {
                out[pos] = prev;
                pos -= 1;
            } else {
                break;
            }
        }
        out[pos] = idx;
    }
    len
}

/// Allocation-free `#![no_std]` 2D `(mean, se)` Pareto-front selector with
/// peeled fronts and deterministic `Key::fold` priority ordering over [`Real`].
pub fn select_pareto<T: Real>(means: &[T], ses: &[T], key: Key, out: &mut [usize]) -> usize {
    let n = means.len().min(ses.len());
    let k = out.len().min(n);
    if k == 0 {
        return 0;
    }
    let mut chosen = 0usize;
    while chosen < k {
        let rem = k - chosen;
        let mut front_len = 0usize;
        for cand in 0..n {
            if out[..chosen].contains(&cand) {
                continue;
            }
            let dom = (0..n).any(|other| {
                other != cand
                    && !out[..chosen].contains(&other)
                    && pareto_dom(means[other], ses[other], means[cand], ses[cand])
            });
            if dom {
                continue;
            }
            let prio = key.fold(cand as u64).seed();
            if front_len == rem {
                let last = out[chosen + front_len - 1];
                let last_prio = key.fold(last as u64).seed();
                if (prio, core::cmp::Reverse(cand)) <= (last_prio, core::cmp::Reverse(last)) {
                    continue;
                }
            } else {
                front_len += 1;
            }
            let mut pos = front_len - 1;
            while pos > 0 {
                let prev = out[chosen + pos - 1];
                let prev_prio = key.fold(prev as u64).seed();
                if (prio, core::cmp::Reverse(cand)) > (prev_prio, core::cmp::Reverse(prev)) {
                    out[chosen + pos] = prev;
                    pos -= 1;
                } else {
                    break;
                }
            }
            out[chosen + pos] = cand;
        }
        if front_len == 0 {
            break;
        }
        chosen += front_len;
    }
    chosen
}

/// Allocation-free `#![no_std]` `NoSurrogate` random distinct candidate
/// selection via `Key::fold`.
pub fn select_random(num_cands: usize, key: Key, out: &mut [usize]) -> usize {
    let k = out.len().min(num_cands);
    if k == 0 {
        return 0;
    }
    let mut len = 0usize;
    for cand in 0..num_cands {
        let prio = key.fold(cand as u64).seed();
        if len == k {
            let last = out[len - 1];
            let last_prio = key.fold(last as u64).seed();
            if (prio, core::cmp::Reverse(cand)) <= (last_prio, core::cmp::Reverse(last)) {
                continue;
            }
        } else {
            len += 1;
        }
        let mut pos = len - 1;
        while pos > 0 {
            let prev = out[pos - 1];
            let prev_prio = key.fold(prev as u64).seed();
            if (prio, core::cmp::Reverse(cand)) > (prev_prio, core::cmp::Reverse(prev)) {
                out[pos] = prev;
                pos -= 1;
            } else {
                break;
            }
        }
        out[pos] = cand;
    }
    len
}

/// Complete surrogate evaluation and candidate selection for a proposal pool.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolSelection {
    pub selected: usize,
    pub initializing: bool,
    pub candidate: baux::semantics::neighbor::query::Prediction,
    pub incumbent: baux::semantics::neighbor::query::Prediction,
    pub baseline_value: f32,
    pub baseline_variance: f32,
    pub baseline_se: f32,
    pub pool_predictions: Vec<(f32, f32, f32)>,
}

/// Type alias matching `SurrogateSpec`.
pub type Surrogate = NeighborValueEstimator;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NeighborEstimatorParams {
    pub(super) neighbors: usize,
    pub(super) epistemic: f64,
    pub(super) aleatoric: f64,
    pub(super) law: UncertaintyLaw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitState {
    pub neighbors: usize,
    pub epistemic: f64,
    pub aleatoric: f64,
    pub scale: f64,
}

/// The bounded implicit-history neighbor estimator used by the Metal
/// pretraining path. Distances arrive from the CUDA implementation of Metal's
/// stratified shadow Gaussian geometry; fitting remains on the host on both
/// backends.
pub struct NeighborValueEstimator {
    pub(super) distances: [[f64; MAX_HISTORY]; MAX_HISTORY],
    pub(super) outcomes: [f32; MAX_HISTORY],
    pub(super) variances: [f32; MAX_HISTORY],
    pub(super) len: usize,
    base: usize,
    identities: [i64; MAX_HISTORY],
    observation: i64,
    fit_candidates: usize,
    fit_samples: usize,
    fit_seed: u64,
    defaults: NeighborEstimatorParams,
    pub(super) params: NeighborEstimatorParams,
    pub(super) y_scale: f64,
    default_scale: f64,
    fit_params: bool,
    adapt_neighbors: bool,
    fitted: bool,
    uncertainty_law: UncertaintyLaw,
    distance_metric: DistanceMetric,
}

impl NeighborValueEstimator {
    pub fn new(initial_value: f32, initial_variance: f32, fit_seed: u64) -> Result<Self, String> {
        Self::configured(initial_value, initial_variance, fit_seed, false, false)
    }

    pub fn configured(
        initial_value: f32,
        initial_variance: f32,
        fit_seed: u64,
        fit_params: bool,
        adapt_neighbors: bool,
    ) -> Result<Self, String> {
        Self::with_spec(
            initial_value,
            initial_variance,
            fit_seed,
            SurrogateSpec {
                fit: fit_params,
                adaptive: adapt_neighbors,
                ..Default::default()
            },
        )
    }

    pub fn with_spec(
        initial_value: f32,
        initial_variance: f32,
        fit_seed: u64,
        spec: SurrogateSpec,
    ) -> Result<Self, String> {
        let spec = spec.validate()?;
        if !initial_value.is_finite() || !initial_variance.is_finite() || initial_variance < 0.0 {
            return Err("neighbor value estimator initial observation is invalid".into());
        }
        let defaults = NeighborEstimatorParams {
            neighbors: spec.neighbors,
            epistemic: f64::from(spec.epistemic),
            aleatoric: f64::from(spec.aleatoric),
            law: UncertaintyLaw::InversePrecision,
        };
        let mut outcomes = [0.0; MAX_HISTORY];
        let mut variances = [0.0; MAX_HISTORY];
        outcomes[0] = initial_value;
        variances[0] = initial_variance;
        Ok(Self {
            distances: [[0.0; MAX_HISTORY]; MAX_HISTORY],
            outcomes,
            variances,
            len: 1,
            base: 0,
            identities: [1; MAX_HISTORY],
            observation: 1,
            fit_candidates: spec.fit_candidates,
            fit_samples: spec.fit_samples,
            fit_seed,
            defaults,
            params: defaults,
            y_scale: f64::from(spec.scale),
            default_scale: f64::from(spec.scale),
            fit_params: spec.fit,
            adapt_neighbors: spec.adaptive,
            fitted: false,
            uncertainty_law: UncertaintyLaw::InversePrecision,
            distance_metric: spec.metric,
        })
    }

    pub fn with_geometry(mut self, geometry: DistanceMetric) -> Self {
        self.distance_metric = geometry;
        self
    }

    pub fn with_law(mut self, law: UncertaintyLaw) -> Self {
        self.uncertainty_law = law;
        self.defaults.law = law;
        self.params.law = law;
        self
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_initializing(&self) -> bool {
        self.len < self.defaults.neighbors
    }

    pub const fn base_slot(&self) -> usize {
        self.base
    }

    pub const fn base_value(&self) -> f32 {
        self.outcomes[self.base]
    }

    pub const fn base_variance(&self) -> f32 {
        self.variances[self.base]
    }

    /// Stable identities used by acquisition draws and bounded retention.
    pub fn identities(&self) -> &[i64] {
        &self.identities[..self.len]
    }

    /// Squared distances in the same slot order as `identities`.
    pub fn distance_row(&self, slot: usize) -> Option<&[f64]> {
        (slot < self.len).then(|| &self.distances[slot][..self.len])
    }

    /// Synchronize the stored pairwise distance matrix after a metric-weight
    /// update (such as learned tensor-family reweighting).
    pub fn sync_distances(&mut self, mut dist: impl FnMut(usize, usize) -> f64) {
        for i in 0..self.len {
            for j in 0..self.len {
                self.distances[i][j] = if i == j { 0.0 } else { dist(i, j).max(0.0) };
            }
        }
        if self.fit_params && self.len >= self.defaults.neighbors {
            let _ = loo_fit::fit(self);
        }
    }

    /// The sampled GPU row and host observation must replace this same slot.
    pub fn next_slot(&self) -> Result<usize, String> {
        if self.len < MAX_HISTORY {
            return Ok(self.len);
        }
        baux::semantics::neighbor::retention(self.identities(), self.identities[self.base])
            .map_err(str::to_string)
    }

    pub const fn fit_state(&self) -> FitState {
        FitState {
            neighbors: self.params.neighbors,
            epistemic: self.params.epistemic,
            aleatoric: self.params.aleatoric,
            scale: self.y_scale,
        }
    }

    pub fn compact(&mut self) {
        let value = self.base_value();
        let variance = self.base_variance();
        let identity = self.identities[self.base];
        self.distances = [[0.0; MAX_HISTORY]; MAX_HISTORY];
        self.outcomes = [0.0; MAX_HISTORY];
        self.variances = [0.0; MAX_HISTORY];
        self.outcomes[0] = value;
        self.variances[0] = variance;
        self.len = 1;
        self.base = 0;
        self.identities[0] = identity;
        self.params = self.defaults;
        self.fitted = false;
        self.y_scale = self.default_scale;
    }

    pub fn incumbent_prediction(&self) -> Result<ValuePrediction, String> {
        let prediction = self.incumbent_baseline()?;
        Ok(ValuePrediction {
            mean: prediction.mean,
            standard_error: prediction.standard_error,
        })
    }

    /// Inclusive incumbent prediction (`excluded = None`) so the incumbent's
    /// own observation at squared distance zero contributes to the
    /// acceptance baseline.
    pub fn incumbent_baseline(
        &self,
    ) -> Result<baux::semantics::neighbor::query::Prediction, String> {
        loo_fit::build_query(
            self,
            &self.distances[self.base][..self.len],
            None,
            0.0,
            None,
            self.distance_metric,
            self.uncertainty_law,
        )
    }

    /// Score all candidate distance rows in a proposal pool and select the
    /// active slot.
    pub fn select_pool(
        &self,
        geometry: &[Vec<f64>],
        step: u32,
        rule: AcquisitionRule,
        forced_slot: Option<usize>,
    ) -> Result<PoolSelection, String> {
        if geometry.is_empty() {
            return Err("invalid candidate pool geometry length".into());
        }
        let pool_count = geometry.len();
        let initializing = self.is_initializing();
        let incumbent = self.incumbent_baseline()?;
        let baseline_value = self.base_value();
        let baseline_variance = self.base_variance().max(0.0);
        let baseline_se = baseline_variance.sqrt();
        let (beta, draws, pareto_seed, random_seed) = match rule {
            AcquisitionRule::Ucb { beta } => (beta, None, None, None),
            AcquisitionRule::Thompson { seed } => {
                let mut buf = [0.0_f32; MAX_HISTORY];
                for (i, &id) in self.identities().iter().enumerate() {
                    buf[i] = baux::semantics::random::normal_metric(seed, id, 0) as f32;
                }
                (0.0, Some(buf), None, None)
            }
            AcquisitionRule::Pareto { seed } => (DEFAULT_BETA, None, Some(seed), None),
            AcquisitionRule::Random { seed } => (0.0, None, None, Some(seed)),
        };
        let mut pool_predictions = Vec::with_capacity(pool_count);
        let mut candidate_preds = Vec::with_capacity(pool_count);
        let mut best_slot = 0usize;
        let mut best_score = BetterScore(f32::NEG_INFINITY);
        for (slot, dist_row) in geometry.iter().enumerate() {
            let pred = self.acquire(dist_row, beta, draws.as_ref().map(|d| &d[..self.len]))?;
            pool_predictions.push((pred.mean, pred.standard_error, pred.score));
            candidate_preds.push(pred);
            let sc = BetterScore(pred.score);
            if sc > best_score {
                best_score = sc;
                best_slot = slot;
            }
        }
        if let Some(seed) = random_seed {
            let mut pick = [0usize; 1];
            if select_random(pool_count, Key::new(seed).fold(u64::from(step)), &mut pick) > 0 {
                best_slot = pick[0];
            }
        } else if let Some(seed) = pareto_seed {
            let mut best_priority = 0_u64;
            let mut found = false;
            for slot in 0..pool_count {
                let pred = candidate_preds[slot];
                let dominated = (0..pool_count).any(|other| {
                    if other == slot {
                        return false;
                    }
                    let o = candidate_preds[other];
                    o.mean >= pred.mean
                        && o.standard_error >= pred.standard_error
                        && (o.mean > pred.mean || o.standard_error > pred.standard_error)
                });
                if !dominated {
                    let priority = baux::semantics::random::splitmix64(
                        seed ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    );
                    if !found || priority > best_priority {
                        found = true;
                        best_priority = priority;
                        best_slot = slot;
                    }
                }
            }
        }
        let selected = if let Some(slot) = forced_slot {
            slot % pool_count
        } else if initializing {
            (step as usize) % pool_count
        } else {
            best_slot
        };
        Ok(PoolSelection {
            selected,
            initializing,
            candidate: candidate_preds[selected],
            incumbent,
            baseline_value,
            baseline_variance,
            baseline_se,
            pool_predictions,
        })
    }

    /// Acquisition-free single-candidate (`K = 1`) surrogate evaluation.
    /// Computes candidate and incumbent predictions directly without pool
    /// acquisition rules (`UCB`, `Thompson`, `Pareto`) or acquisition RNG
    /// draws.
    pub fn evaluate_singleton(&self, distances: &[f64]) -> Result<PoolSelection, String> {
        let initializing = self.is_initializing();
        let incumbent = self.incumbent_baseline()?;
        let baseline_value = self.base_value();
        let baseline_variance = self.base_variance().max(0.0);
        let baseline_se = baseline_variance.sqrt();
        let candidate = self.acquire(distances, 0.0, None)?;
        Ok(PoolSelection {
            selected: 0,
            initializing,
            candidate,
            incumbent,
            baseline_value,
            baseline_variance,
            baseline_se,
            pool_predictions: vec![(candidate.mean, candidate.standard_error, candidate.mean)],
        })
    }

    pub fn predict(&self, distances: &[f64]) -> Result<ValuePrediction, String> {
        self.predict_law(distances, self.uncertainty_law)
    }

    /// Predict using upper-half-space hyperbolic ENN (Poincaré 1882; Beardon
    /// 1983; Cannon et al. 1997), setting each observation's vertical
    /// coordinate `z_i` to its local k-NN Euclidean scale
    /// `sqrt(D_{i,(k)})`.
    pub fn predict_hyperbolic(
        &self,
        distances: &[f64],
        local_k: usize,
    ) -> Result<ValuePrediction, String> {
        if distances.len() != self.len || local_k == 0 {
            return Err("invalid hyperbolic query history length or local k".into());
        }
        let prediction = loo_fit::build_query(
            self,
            distances,
            None,
            0.0,
            None,
            DistanceMetric::Hyperbolic { local_k },
            self.uncertainty_law,
        )?;
        Ok(ValuePrediction {
            mean: prediction.mean,
            standard_error: prediction.standard_error,
        })
    }

    pub fn predict_law(
        &self,
        distances: &[f64],
        law: UncertaintyLaw,
    ) -> Result<ValuePrediction, String> {
        let prediction =
            loo_fit::build_query(self, distances, None, 0.0, None, self.distance_metric, law)?;
        Ok(ValuePrediction {
            mean: prediction.mean,
            standard_error: prediction.standard_error,
        })
    }

    pub fn acquire(
        &self,
        distances: &[f64],
        beta: f32,
        draws: Option<&[f32]>,
    ) -> Result<baux::semantics::neighbor::query::Prediction, String> {
        loo_fit::build_query(
            self,
            distances,
            None,
            beta,
            draws,
            self.distance_metric,
            self.uncertainty_law,
        )
    }

    /// Append a new observation and return the archive slot index that was
    /// written.
    pub fn append(
        &mut self,
        row: &[f64],
        value: f32,
        variance: f32,
        accepted: bool,
    ) -> Result<usize, String> {
        if row.len() != self.len
            || row
                .iter()
                .any(|distance| !distance.is_finite() || *distance < 0.0)
            || !value.is_finite()
            || !variance.is_finite()
            || variance < 0.0
        {
            return Err(
                "neighbor value estimator observation or sampled distance row is invalid".into(),
            );
        }
        let next = self.next_slot()?;
        let identity = self
            .observation
            .checked_add(1)
            .ok_or("observation identity overflow")?;
        for (previous, &distance) in row.iter().enumerate() {
            self.distances[next][previous] = distance;
            self.distances[previous][next] = distance;
        }
        self.outcomes[next] = value;
        self.variances[next] = variance;
        self.distances[next][next] = 0.0;
        self.identities[next] = identity;
        self.observation = identity;
        self.len = (self.len + 1).min(MAX_HISTORY);
        if accepted {
            self.base = next;
        }
        if self.fit_params && self.len >= self.defaults.neighbors {
            loo_fit::fit(self)?;
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_eval() {
        let estimator = NeighborValueEstimator::new(1.25, 0.04, 73).unwrap();
        let selection = estimator.evaluate_singleton(&[0.0_f64]).unwrap();
        assert_eq!(selection.selected, 0);
        assert!(selection.initializing);
        let pred = estimator.predict(&[0.0_f64]).unwrap();
        assert_eq!(selection.candidate.mean, pred.mean);
        assert_eq!(selection.candidate.standard_error, pred.standard_error);
        assert_eq!(selection.candidate.score, pred.mean);
        assert_eq!(selection.pool_predictions.len(), 1);
        assert_eq!(
            selection.pool_predictions[0],
            (pred.mean, pred.standard_error, pred.mean)
        );
    }
}
