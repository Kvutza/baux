use super::{
    DistanceMetric, MAX_HISTORY, NeighborEstimatorParams, NeighborValueEstimator, UncertaintyLaw,
};
use rand::{SeedableRng, rngs::StdRng};

pub(super) fn metric_row(
    est: &NeighborValueEstimator,
    distances: &[f64],
    excluded: Option<usize>,
    metric: DistanceMetric,
) -> [f64; MAX_HISTORY] {
    let mut row_dists = [0.0_f64; MAX_HISTORY];
    let n = distances.len().min(est.len);
    row_dists[..n].copy_from_slice(&distances[..n]);
    let local_k = match metric {
        DistanceMetric::Euclidean => return row_dists,
        DistanceMetric::Conformal { local_k }
        | DistanceMetric::Hyperbolic { local_k }
        | DistanceMetric::Wasserstein { local_k } => {
            if local_k == 0 || n <= 1 {
                return row_dists;
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
    let scales = baux::semantics::neighbor::local_scales(n, local_k, |r, c| est.distances[r][c]);
    for slot in 0..n {
        let sigma_i = scales[slot].sqrt();
        row_dists[slot] =
            baux::semantics::neighbor::metric_delta(metric, distances[slot], sigma_q, sigma_i);
    }
    row_dists
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
    let row_dists = metric_row(est, distances, excluded, metric);
    let mut pred = baux::semantics::neighbor::query::Query {
        distances: &row_dists[..est.len],
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

fn characteristic_distance(est: &NeighborValueEstimator) -> f64 {
    let n = est.len;
    if est.distance_metric == DistanceMetric::Euclidean {
        return baux::semantics::neighbor::characteristic_distance(n, |row, col| {
            est.distances[row][col]
        });
    }
    let mut rows = [[0.0_f64; MAX_HISTORY]; MAX_HISTORY];
    for (row, dst) in rows.iter_mut().enumerate().take(n) {
        *dst = metric_row(
            est,
            &est.distances[row][..n],
            Some(row),
            est.distance_metric,
        );
    }
    baux::semantics::neighbor::characteristic_distance(n, |row, col| rows[row][col])
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
    let d_scale = characteristic_distance(est);
    let mut rng = StdRng::seed_from_u64(est.fit_seed.wrapping_add(n as u64));
    let mut candidates = Vec::with_capacity(est.fit_candidates + 1);
    for (epistemic, aleatoric) in
        baux::semantics::neighbor::draw_scales(est.fit_candidates, d_scale, true, &mut rng)
    {
        candidates.push(NeighborEstimatorParams {
            neighbors: est.defaults.neighbors,
            epistemic,
            aleatoric,
            law: est.uncertainty_law,
        });
    }
    candidates.push(NeighborEstimatorParams {
        neighbors: est.defaults.neighbors,
        ..est.params
    });
    let selected = baux::semantics::neighbor::sampled_rows(n, est.fit_samples, &mut rng);
    let mut best: Option<(NeighborEstimatorParams, f64)> = None;
    for candidate in candidates {
        let score = loo_score(est, candidate, &selected);
        if score.is_finite() && best.is_none_or(|(_, best_score)| score >= best_score) {
            best = Some((candidate, score));
        }
    }
    let mut fitted = best
        .ok_or("neighbor value estimator scale fitting produced no finite candidate")?
        .0;
    if est.adapt_neighbors && n > 2 {
        let selected_k = baux::semantics::neighbor::sampled_rows(n, est.fit_samples, &mut rng);
        let largest = est.defaults.neighbors.min(n - 1);
        let mut best_k: Option<(NeighborEstimatorParams, f64)> = None;
        for k in 1..=largest {
            let candidate = NeighborEstimatorParams {
                neighbors: k,
                ..fitted
            };
            let score = loo_score(est, candidate, &selected_k);
            if score.is_finite() && best_k.is_none_or(|(_, best_score)| score > best_score) {
                best_k = Some((candidate, score));
            }
        }
        fitted = best_k
            .ok_or("neighbor value estimator adaptive k fitting produced no finite candidate")?
            .0;
    }
    est.params = fitted;
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
        let row_dists = metric_row(
            est,
            &est.distances[held][..est.len],
            Some(held),
            est.distance_metric,
        );
        let (mean, standard_error, _, _) = baux::semantics::neighbor::query::posterior_f64(
            &row_dists[..est.len],
            &est.outcomes[..est.len],
            &est.variances[..est.len],
            params.neighbors,
            params.epistemic,
            params.aleatoric,
            est.y_scale,
            params.law,
            Some(held),
            false,
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
        let (mean, se, _, _) = baux::semantics::neighbor::query::posterior_f64(
            &[d0],
            &[y0],
            &[v0],
            1,
            0.05,
            0.05,
            1.0,
            UncertaintyLaw::BiasVariance,
            None,
            false,
        );
        assert!(mean.is_finite());
        assert!(se.is_finite() && se > 0.0);
    }
}
