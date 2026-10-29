use super::{
    DistanceMetric, MAX_HISTORY, NeighborEstimatorParams, NeighborValueEstimator, UncertaintyLaw,
};
use rand::seq::index::sample;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub(super) fn scale_row(
    est: &NeighborValueEstimator,
    distances: &[f64],
    excluded: Option<usize>,
    metric: DistanceMetric,
) -> [f64; MAX_HISTORY] {
    let mut scaled = [0.0_f64; MAX_HISTORY];
    let n = distances.len().min(est.len);
    scaled[..n].copy_from_slice(&distances[..n]);
    let local_k = match metric {
        DistanceMetric::Euclidean => return scaled,
        DistanceMetric::SelfTuning { local_k } | DistanceMetric::Hyperbolic { local_k } => {
            if local_k == 0 || n <= 1 {
                return scaled;
            }
            local_k
        }
    };
    let mut query_positive = [0.0_f64; MAX_HISTORY];
    let mut qp_len = 0;
    for (idx, &d) in distances[..n].iter().enumerate() {
        if Some(idx) != excluded && d > 0.0 {
            query_positive[qp_len] = d;
            qp_len += 1;
        }
    }
    let sigma_q = if qp_len == 0 {
        1.0
    } else {
        query_positive[..qp_len].sort_unstable_by(f64::total_cmp);
        query_positive[local_k.min(qp_len) - 1].max(1.0e-12).sqrt()
    };
    for slot in 0..n {
        let mut row = [0.0_f64; MAX_HISTORY];
        let mut row_len = 0;
        for col in 0..n {
            let d = est.distances[slot][col];
            if col != slot && d > 0.0 {
                row[row_len] = d;
                row_len += 1;
            }
        }
        let sigma_i = if row_len == 0 {
            sigma_q
        } else {
            row[..row_len].sort_unstable_by(f64::total_cmp);
            row[local_k.min(row_len) - 1].max(1.0e-12).sqrt()
        };
        scaled[slot] = match metric {
            DistanceMetric::Euclidean => distances[slot],
            DistanceMetric::SelfTuning { .. } => {
                baux::semantics::neighbor::selftuning_delta(distances[slot], sigma_q, sigma_i)
            }
            DistanceMetric::Hyperbolic { .. } => {
                baux::semantics::neighbor::halfspace_delta(distances[slot], sigma_q, sigma_i)
            }
        };
    }
    scaled
}

fn effective_scale(est: &NeighborValueEstimator) -> f32 {
    (est.y_scale as f32).max(1.0e-6)
}

pub(super) fn build_query(
    est: &NeighborValueEstimator,
    distances: &[f64],
    excluded: Option<usize>,
    beta: f32,
    draws: Option<&[f32]>,
    metric: DistanceMetric,
    law: UncertaintyLaw,
) -> Result<baux::semantics::neighbor::query::Prediction, String> {
    if distances.len() != est.len
        || distances
            .iter()
            .any(|distance| !distance.is_finite() || *distance < 0.0)
    {
        return Err("invalid query history length".into());
    }
    let params = if est.fitted { est.params } else { est.defaults };
    let scale = effective_scale(est);
    let b = est.base;
    let y_b = est.outcomes[b];
    let mut base_deltas = [0.0_f32; MAX_HISTORY];
    for i in 0..est.len {
        base_deltas[i] = est.outcomes[i] - y_b;
    }
    let scaled = scale_row(est, distances, excluded, metric);
    let mut pred = baux::semantics::neighbor::query::Query {
        distances: &scaled[..est.len],
        outcomes: &base_deltas[..est.len],
        variances: &est.variances[..est.len],
        draws,
        neighbors: params.neighbors,
        epistemic: params.epistemic as f32,
        aleatoric: params.aleatoric as f32,
        scale,
        beta,
        excluded,
        incumbent: b,
        adaptive: est.adapt_neighbors,
    }
    .predict_law(law)
    .map_err(str::to_string)?;
    pred.mean += y_b;
    pred.score += y_b;
    Ok(pred)
}

pub(super) fn fit(est: &mut NeighborValueEstimator) -> Result<(), String> {
    let n = est.len;
    let sum = est.outcomes[..n]
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>();
    let sumsq = est.outcomes[..n]
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    let mean = sum / n as f64;
    let variance = (sumsq / n as f64 - mean * mean).max(0.0);
    est.y_scale = variance.sqrt();
    if !est.y_scale.is_finite() || est.y_scale <= 0.0 {
        est.y_scale = 1.0;
    }
    let mut rng = StdRng::seed_from_u64(est.fit_seed.wrapping_add(n as u64));
    let mut candidates = Vec::with_capacity(est.fit_candidates + 1);
    let epistemic = (0..est.fit_candidates)
        .map(|_| 10.0_f64.powf(rng.gen_range(-3.0..=3.0)))
        .collect::<Vec<_>>();
    let aleatoric = (0..est.fit_candidates)
        .map(|_| 10.0_f64.powf(rng.gen_range(-3.0..=3.0)))
        .collect::<Vec<_>>();
    for (&epistemic, &aleatoric) in epistemic.iter().zip(&aleatoric) {
        candidates.push(NeighborEstimatorParams {
            neighbors: est.params.neighbors,
            epistemic,
            aleatoric,
            law: est.uncertainty_law,
        });
    }
    candidates.push(est.params);
    let selected = sampled_rows(n, est.fit_samples, &mut rng);
    let mut best: Option<(NeighborEstimatorParams, f64)> = None;
    for candidate in candidates {
        let score = loo_score(est, candidate, &selected);
        if score.is_finite() && best.is_none_or(|(_, best_score)| score >= best_score) {
            best = Some((candidate, score));
        }
    }
    est.params = best
        .ok_or("neighbor value estimator scale fitting produced no finite candidate")?
        .0;
    est.fitted = true;
    Ok(())
}

fn loo_score(
    est: &NeighborValueEstimator,
    params: NeighborEstimatorParams,
    selected: &[usize],
) -> f64 {
    let mut score = 0.0;
    for &held in selected {
        let scaled = scale_row(
            est,
            &est.distances[held][..est.len],
            Some(held),
            est.distance_metric,
        );
        let (mean, standard_error) = posterior(
            &scaled[..est.len],
            &est.outcomes[..est.len],
            &est.variances[..est.len],
            params,
            est.y_scale,
            Some(held),
        );
        if !mean.is_finite() || !standard_error.is_finite() || standard_error <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let target = f64::from(est.outcomes[held]) / est.y_scale;
        let mean = mean / est.y_scale;
        let standard_error = standard_error / est.y_scale;
        let residual = (target - mean) / standard_error;
        score += -0.5 * (2.0 * std::f64::consts::PI).ln()
            - standard_error.ln()
            - 0.5 * residual * residual;
    }
    score
}

fn sampled_rows(n: usize, count: usize, rng: &mut StdRng) -> Vec<usize> {
    if count >= n {
        (0..n).collect()
    } else {
        sample(rng, n, count).into_iter().collect()
    }
}

/// Zero-allocation FP64 leave-one-out posterior for hyperparameter scale
/// fitting.
fn posterior(
    distances: &[f64],
    outcomes: &[f32],
    variances: &[f32],
    params: NeighborEstimatorParams,
    y_scale: f64,
    excluded: Option<usize>,
) -> (f64, f64) {
    let mut nearest = [0usize; MAX_HISTORY];
    let mut count = 0;
    for i in 0..distances.len() {
        if Some(i) == excluded {
            continue;
        }
        let mut pos = count;
        while pos > 0 && distances[i] < distances[nearest[pos - 1]] {
            nearest[pos] = nearest[pos - 1];
            pos -= 1;
        }
        nearest[pos] = i;
        count += 1;
    }
    count = count.min(params.neighbors);
    if count == 0 {
        let fallback = excluded.unwrap_or(0).min(outcomes.len() - 1);
        return (
            f64::from(outcomes[fallback]),
            params.aleatoric.max(1.0e-12).sqrt() * y_scale,
        );
    }
    let scale_squared = (y_scale * y_scale).max(1.0e-12);
    let mut weights = [0.0_f64; MAX_HISTORY];
    let mut total = 0.0_f64;
    for (slot, &i) in nearest[..count].iter().enumerate() {
        let w = 1.0
            / (1.0e-9
                + params.epistemic * distances[i]
                + params.aleatoric
                + f64::from(variances[i]) / scale_squared);
        weights[slot] = w;
        total += w;
    }
    let total = total.max(1.0e-12);
    let mut mean = 0.0_f64;
    let mut dist_var = 0.0_f64;
    for (slot, &i) in nearest[..count].iter().enumerate() {
        let w = weights[slot];
        mean += w * f64::from(outcomes[i]);
        dist_var += (w / total) * distances[i];
    }
    mean /= total;
    let var = match params.law {
        UncertaintyLaw::Geometric => params.epistemic * dist_var,
        UncertaintyLaw::BiasVariance => {
            let mut wn = 0.0_f64;
            for (slot, &i) in nearest[..count].iter().enumerate() {
                let w = weights[slot];
                wn += w * w * (params.aleatoric + f64::from(variances[i]) / scale_squared);
            }
            wn / (total * total).max(1.0e-24) + params.epistemic * dist_var
        }
        UncertaintyLaw::InversePrecision => {
            let mut al = 0.0_f64;
            for (slot, &i) in nearest[..count].iter().enumerate() {
                let w = weights[slot];
                al += (w / total) * (params.aleatoric + f64::from(variances[i]) / scale_squared);
            }
            1.0 / total + al
        }
    };
    (mean, var.max(1.0e-12).sqrt() * y_scale)
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn prove_posterior() {
        let d0: f64 = kani::any();
        let y0: f32 = kani::any();
        let v0: f32 = kani::any();
        kani::assume(d0.is_finite() && (0.0..=1.0e6).contains(&d0));
        kani::assume(y0.is_finite() && (-1.0e3..=1.0e3).contains(&y0));
        kani::assume(v0.is_finite() && (0.0..=1.0e3).contains(&v0));
        let params = NeighborEstimatorParams {
            neighbors: 1,
            epistemic: 0.05,
            aleatoric: 0.05,
            law: UncertaintyLaw::BiasVariance,
        };
        let (mean, se) = posterior(&[d0], &[y0], &[v0], params, 1.0, None);
        assert!(mean.is_finite());
        assert!(se.is_finite() && se > 0.0);
    }
}
