//! `#![no_std]` precision-weighted ENN posterior and acquisition primitives
//! parameterized over generic [`Real`] scalars (`f32` and `f64`).
//!
//! Order of accumulation and first-index tie breaking are part of this rule.

use super::{DistanceMetric, MAX_HISTORY, MetricKind, Real, TopK, UncertaintyLaw, eval_metric};

/// Full `#![no_std]` posterior decomposition `(mu, se_total, se_epi, se_ale,
/// count)` parameterized over [`Real`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Posterior<T: Real = f64> {
    pub mu: T,
    pub se_total: T,
    pub se_epi: T,
    pub se_ale: T,
    pub count: usize,
}

pub struct Query<'a> {
    pub distances: &'a [f64],
    pub outcomes: &'a [f32],
    pub variances: &'a [f32],
    pub draws: Option<&'a [f32]>,
    pub neighbors: usize,
    pub epistemic: f32,
    pub aleatoric: f32,
    pub scale: f32,
    pub beta: f32,
    pub excluded: Option<usize>,
    pub incumbent: usize,
    pub adaptive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub mean: f32,
    /// Total predictive standard error, including the observation-noise term.
    pub standard_error: f32,
    /// Epistemic standard error used by UCB and the sampled function.
    pub epistemic_error: f32,
    pub score: f32,
    pub neighbors: usize,
}

/// Per-neighbor precision weight `1 / max(PREC_EPS + epi * dist + ale +
/// yvar_scaled, NOISE_FLOOR)` over [`Real`].
#[inline(always)]
pub fn neighbor_weight<T: Real>(dist: T, epistemic: T, aleatoric: T, yvar_scaled: T) -> T {
    let total = T::PREC_EPS + epistemic * dist + aleatoric + yvar_scaled;
    T::ONE / total.max(T::NOISE_FLOOR)
}

/// `ConstParamTy` posterior error law returning `(se_total, se_epi, se_ale)`
/// over [`Real`].
#[inline(always)]
pub fn error_law<const L: UncertaintyLaw, T: Real>(
    denom: T,
    inv_denom: T,
    weighted_noise: T,
    sq_noise: T,
    dist_var: T,
    epistemic: T,
    scale: T,
) -> (T, T, T) {
    match L {
        UncertaintyLaw::InversePrecision => {
            let total = (inv_denom + weighted_noise).max(T::NOISE_FLOOR).sqrt() * scale;
            let epi = inv_denom.max(T::ZERO).sqrt() * scale;
            let ale = weighted_noise.max(T::ZERO).sqrt() * scale;
            (total, epi, ale)
        }
        UncertaintyLaw::BiasVariance => {
            let geom_var = epistemic * dist_var;
            let noise_var = sq_noise / (denom * denom).max(T::VAR_FLOOR);
            let total = (noise_var + geom_var).max(T::NOISE_FLOOR).sqrt() * scale;
            let epi = geom_var.max(T::ZERO).sqrt() * scale;
            let ale = noise_var.max(T::ZERO).sqrt() * scale;
            (total, epi, ale)
        }
        UncertaintyLaw::Geometric => {
            let se = (epistemic * dist_var).max(T::NOISE_FLOOR).sqrt() * scale;
            (se, se, T::ZERO)
        }
    }
}

/// Evaluate `#![no_std]` ENN posterior directly from a sorted [`TopK`]
/// `(distance, index)` slice and optional heteroscedastic `variances` (`yvar`).
#[inline]
pub fn posterior_topk<const L: UncertaintyLaw, T: Real>(
    topk: &[(T, u32)],
    outcomes: &[T],
    variances: Option<&[T]>,
    epistemic: T,
    aleatoric: T,
    scale: T,
    adaptive: bool,
) -> Posterior<T> {
    let mut count = topk.len();
    if count == 0 {
        let fallback = outcomes.first().copied().unwrap_or(T::ZERO);
        let ale = aleatoric.max(T::NOISE_FLOOR).sqrt() * scale;
        return Posterior {
            mu: fallback,
            se_total: ale,
            se_epi: T::ZERO,
            se_ale: ale,
            count: 0,
        };
    }
    let scale_sq = (scale * scale).max(T::NOISE_FLOOR);
    let noise_at = |idx: usize| -> T {
        let yv = variances
            .and_then(|v| v.get(idx))
            .copied()
            .unwrap_or(T::ZERO)
            .max(T::ZERO);
        aleatoric + yv / scale_sq
    };
    if adaptive {
        let mut p_sum = T::ZERO;
        let mut n_sum = T::ZERO;
        let mut best_risk = T::INFINITY;
        let mut chosen = 1usize;
        for (rank, &(dist, idx)) in topk.iter().enumerate() {
            let nz = noise_at(idx as usize);
            let p = neighbor_weight(dist, epistemic, nz, T::ZERO);
            p_sum += p;
            n_sum += p * p * nz;
            let v_err = n_sum / (p_sum * p_sum).max(T::VAR_FLOOR);
            let risk = epistemic * dist + v_err;
            if risk < best_risk {
                best_risk = risk;
                chosen = rank + 1;
            }
        }
        count = chosen.clamp(1, topk.len());
    }
    let used = &topk[..count];
    let mut denom = T::ZERO;
    for &(dist, idx) in used {
        denom += neighbor_weight(dist, epistemic, noise_at(idx as usize), T::ZERO);
    }
    let denom_clamped = denom.max(T::NOISE_FLOOR);
    let inv_denom = T::ONE / denom_clamped;
    let mut mu = T::ZERO;
    let mut weighted_noise = T::ZERO;
    let mut sq_noise = T::ZERO;
    let mut dist_var = T::ZERO;
    for &(dist, idx) in used {
        let i = idx as usize;
        let nz = noise_at(i);
        let w = neighbor_weight(dist, epistemic, nz, T::ZERO);
        let norm_w = w * inv_denom;
        mu += norm_w * outcomes[i];
        weighted_noise += norm_w * nz;
        sq_noise += w * w * nz;
        dist_var += norm_w * dist;
    }
    let (se_total, se_epi, se_ale) = error_law::<L, T>(
        denom_clamped,
        inv_denom,
        weighted_noise,
        sq_noise,
        dist_var,
        epistemic,
        scale,
    );
    Posterior {
        mu,
        se_total,
        se_epi,
        se_ale,
        count,
    }
}

/// Compile-time specialized (`MetricKind` + `UncertaintyLaw`) `#![no_std]`
/// slice posterior over up to `MAX_HISTORY` resident observations.
///
/// Both `query_scale` and `local_scales[i]` are squared `k`-NN distances
/// `d_k^2`, converted internally to linear radii `sigma = sqrt(max(d_k^2,
/// NOISE_FLOOR))`.
#[allow(clippy::too_many_arguments)]
pub fn posterior_const<const M: MetricKind, const L: UncertaintyLaw, T: Real>(
    distances: &[T],
    local_scales: Option<&[T]>,
    query_scale: T,
    outcomes: &[T],
    variances: Option<&[T]>,
    neighbors: usize,
    epistemic: T,
    aleatoric: T,
    scale: T,
    excluded: Option<usize>,
    adaptive: bool,
) -> Posterior<T> {
    let n = distances.len().min(outcomes.len()).min(MAX_HISTORY);
    let mut topk = TopK::<T, MAX_HISTORY>::new(neighbors.min(MAX_HISTORY));
    let s_q = query_scale.max(T::NOISE_FLOOR).sqrt();
    for (i, &raw_d) in distances.iter().enumerate().take(n) {
        if Some(i) == excluded {
            continue;
        }
        let s_h = local_scales
            .and_then(|s| s.get(i))
            .copied()
            .unwrap_or(T::ONE)
            .max(T::NOISE_FLOOR)
            .sqrt();
        let d = eval_metric::<M, T>(raw_d, s_q, s_h);
        topk.push(d, i as u32);
    }
    if topk.is_empty() {
        let fallback = excluded.unwrap_or(0).min(outcomes.len().saturating_sub(1));
        let ale = aleatoric.max(T::NOISE_FLOOR).sqrt() * scale;
        return Posterior {
            mu: outcomes.get(fallback).copied().unwrap_or(T::ZERO),
            se_total: ale,
            se_epi: T::ZERO,
            se_ale: ale,
            count: 0,
        };
    }
    posterior_topk::<L, T>(
        topk.as_slice(),
        &outcomes[..n],
        variances.map(|v| &v[..v.len().min(n)]),
        epistemic,
        aleatoric,
        scale,
        adaptive,
    )
}

/// Runtime-dispatched counterpart to [`posterior_const`] over `(DistanceMetric,
/// UncertaintyLaw)`.
#[allow(clippy::too_many_arguments)]
pub fn posterior_eval<T: Real>(
    metric: DistanceMetric,
    law: UncertaintyLaw,
    distances: &[T],
    local_scales: Option<&[T]>,
    query_scale: T,
    outcomes: &[T],
    variances: Option<&[T]>,
    neighbors: usize,
    epistemic: T,
    aleatoric: T,
    scale: T,
    excluded: Option<usize>,
    adaptive: bool,
) -> Posterior<T> {
    match (MetricKind::from_metric(metric), law) {
        (MetricKind::Euclidean, UncertaintyLaw::InversePrecision) => {
            posterior_const::<{ MetricKind::Euclidean }, { UncertaintyLaw::InversePrecision }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Euclidean, UncertaintyLaw::BiasVariance) => {
            posterior_const::<{ MetricKind::Euclidean }, { UncertaintyLaw::BiasVariance }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Euclidean, UncertaintyLaw::Geometric) => {
            posterior_const::<{ MetricKind::Euclidean }, { UncertaintyLaw::Geometric }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Conformal, UncertaintyLaw::InversePrecision) => {
            posterior_const::<{ MetricKind::Conformal }, { UncertaintyLaw::InversePrecision }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Conformal, UncertaintyLaw::BiasVariance) => {
            posterior_const::<{ MetricKind::Conformal }, { UncertaintyLaw::BiasVariance }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Conformal, UncertaintyLaw::Geometric) => {
            posterior_const::<{ MetricKind::Conformal }, { UncertaintyLaw::Geometric }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Hyperbolic, UncertaintyLaw::InversePrecision) => {
            posterior_const::<{ MetricKind::Hyperbolic }, { UncertaintyLaw::InversePrecision }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Hyperbolic, UncertaintyLaw::BiasVariance) => {
            posterior_const::<{ MetricKind::Hyperbolic }, { UncertaintyLaw::BiasVariance }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Hyperbolic, UncertaintyLaw::Geometric) => {
            posterior_const::<{ MetricKind::Hyperbolic }, { UncertaintyLaw::Geometric }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Wasserstein, UncertaintyLaw::InversePrecision) => {
            posterior_const::<{ MetricKind::Wasserstein }, { UncertaintyLaw::InversePrecision }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Wasserstein, UncertaintyLaw::BiasVariance) => {
            posterior_const::<{ MetricKind::Wasserstein }, { UncertaintyLaw::BiasVariance }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
        (MetricKind::Wasserstein, UncertaintyLaw::Geometric) => {
            posterior_const::<{ MetricKind::Wasserstein }, { UncertaintyLaw::Geometric }, T>(
                distances,
                local_scales,
                query_scale,
                outcomes,
                variances,
                neighbors,
                epistemic,
                aleatoric,
                scale,
                excluded,
                adaptive,
            )
        }
    }
}

fn nearest_indices(
    distances: &[f64],
    neighbors: usize,
    excluded: Option<usize>,
) -> ([usize; MAX_HISTORY], usize) {
    let mut nearest = [0; MAX_HISTORY];
    let mut count = 0;
    for i in 0..distances.len() {
        if Some(i) == excluded {
            continue;
        }
        let mut position = count;
        while position > 0 && distances[i] < distances[nearest[position - 1]] {
            nearest[position] = nearest[position - 1];
            position -= 1;
        }
        nearest[position] = i;
        count += 1;
    }
    (nearest, count.min(neighbors))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn query_posterior(
    distances: &[f64],
    outcomes: &[f32],
    variances: &[f32],
    neighbors: usize,
    epistemic: f64,
    aleatoric: f64,
    scale: f64,
    law: UncertaintyLaw,
    excluded: Option<usize>,
    adaptive: bool,
) -> Posterior<f64> {
    let n = distances
        .len()
        .min(outcomes.len())
        .min(variances.len())
        .min(MAX_HISTORY);
    let mut out_buf = [0.0; MAX_HISTORY];
    let mut var_buf = [0.0; MAX_HISTORY];
    for i in 0..n {
        out_buf[i] = f64::from(outcomes[i]);
        var_buf[i] = f64::from(variances[i]);
    }
    posterior_eval::<f64>(
        DistanceMetric::Euclidean,
        law,
        &distances[..n],
        None,
        1.0,
        &out_buf[..n],
        Some(&var_buf[..n]),
        neighbors,
        epistemic,
        aleatoric,
        scale,
        excluded,
        adaptive,
    )
}

impl Query<'_> {
    pub fn predict(&self) -> Result<Prediction, &'static str> {
        self.predict_law(UncertaintyLaw::InversePrecision)
    }

    fn validate(&self) -> Result<(), &'static str> {
        let len = self.distances.len();
        if len == 0
            || len > MAX_HISTORY
            || self.outcomes.len() != len
            || self.variances.len() != len
            || self.neighbors == 0
            || self.incumbent >= len
            || self.excluded.is_some_and(|i| i >= len)
            || self
                .draws
                .is_some_and(|d| d.len() != len || d.iter().any(|v| !v.is_finite()))
            || self.distances.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.variances.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.outcomes.iter().any(|v| !v.is_finite())
            || [self.epistemic, self.aleatoric, self.scale, self.beta]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            Err("invalid resident posterior query")
        } else {
            Ok(())
        }
    }

    fn precision(&self, i: usize) -> f64 {
        let scale_squared = (f64::from(self.scale) * f64::from(self.scale)).max(1.0e-12);
        1.0 / (1.0e-9
            + f64::from(self.epistemic) * self.distances[i]
            + f64::from(self.aleatoric)
            + f64::from(self.variances[i]) / scale_squared)
            .max(1.0e-12)
    }

    pub fn predict_law(&self, law: UncertaintyLaw) -> Result<Prediction, &'static str> {
        self.validate()?;
        let post = query_posterior(
            self.distances,
            self.outcomes,
            self.variances,
            self.neighbors,
            f64::from(self.epistemic),
            f64::from(self.aleatoric),
            f64::from(self.scale),
            law,
            self.excluded,
            self.adaptive,
        );
        let (mean, error, epistemic, count) = (post.mu, post.se_total, post.se_epi, post.count);
        if count == 0 {
            let mean = self.outcomes[self.excluded.unwrap_or(self.incumbent)];
            return Ok(Prediction {
                mean,
                standard_error: error as f32,
                epistemic_error: 0.0,
                score: mean,
                neighbors: 0,
            });
        }
        let score = if let Some(draws) = self.draws {
            let (nearest, _) = nearest_indices(self.distances, self.neighbors, self.excluded);
            let slice = &nearest[..count];
            let mut reference = f64::MIN_POSITIVE;
            for &i in slice {
                reference = reference.max(self.precision(i));
            }
            let (mut noise, mut squared) = (0.0, 0.0);
            for &i in slice {
                let weight = self.precision(i) / reference;
                noise += weight * f64::from(draws[i]);
                squared += weight * weight;
            }
            mean + epistemic * (noise / squared.sqrt().max(1.0e-12))
        } else {
            mean + f64::from(self.beta) * epistemic
        };
        if [mean, error, epistemic, score]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("resident posterior arithmetic overflow");
        }
        Ok(Prediction {
            mean: mean as f32,
            standard_error: error as f32,
            epistemic_error: epistemic as f32,
            score: score as f32,
            neighbors: count,
        })
    }
}
