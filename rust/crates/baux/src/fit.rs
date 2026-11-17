//! Parameter fitting for ENN models via subsample log-likelihood.

use std::collections::HashSet;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use crate::error::ENNError;
use crate::hash::Key;
use crate::model::ENN;
use crate::params::ENNParams;
use crate::posterior::{WeightedPosteriorData, compute_posterior, index_search};

fn metric_distances(
    distances: &ArrayView2<f64>,
    k: usize,
    metric: baux::semantics::neighbor::DistanceMetric,
    label: &str,
) -> Result<Array2<f64>, ENNError> {
    let n = distances.nrows();
    if distances.ncols() != n || k == 0 {
        return Err(ENNError::InvalidParameter(format!(
            "{label} distances require a square matrix and positive local k"
        )));
    }
    if distances
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ENNError::InvalidParameter(format!(
            "{label} distances must be finite and nonnegative"
        )));
    }
    if n <= 1 {
        return Ok(distances.to_owned());
    }
    let scales = baux::semantics::neighbor::local_scales(n, k, |row, col| distances[[row, col]]);
    Ok(Array2::from_shape_fn((n, n), |(row, column)| {
        if row == column {
            0.0
        } else {
            baux::semantics::neighbor::metric_delta(
                metric,
                distances[[row, column]],
                scales[row].sqrt(),
                scales[column].sqrt(),
            )
        }
    }))
}

/// Conformally normalize a square matrix of squared distances by per-row local
/// radii (Zelnik-Manor and Perona, 2004, *Self-Tuning Spectral Clustering*,
/// NeurIPS 17, pp. 1601-1608).
///
/// Each radius is the `k`th positive off-diagonal Euclidean `k`-NN radius
/// `sigma_i = sqrt(d_k^2)`. The geometric mean keeps `sq_dist / (scale_a *
/// scale_b)` dimensionless while preserving symmetry.
pub fn conformal_distances(distances: &ArrayView2<f64>, k: usize) -> Result<Array2<f64>, ENNError> {
    metric_distances(
        distances,
        k,
        baux::semantics::neighbor::DistanceMetric::Conformal { local_k: k },
        "conformal",
    )
}

/// Upper half-space Poincaré hyperbolic dissimilarity matrix
/// (Poincaré, *Acta Math.* 1:1-62, 1882; Beardon, *The Geometry of Discrete
/// Groups*, Springer GTM 91, 1983, sec. 7.2, Theorem 7.2.1; Cannon, Floyd,
/// Kenyon, and Parry, *Hyperbolic Geometry*, MSRI Publ. 31, 1997).
///
/// Computes `(sq_dist + (scale_a - scale_b)^2) / (2 * scale_a * scale_b)
/// = cosh(d_H((x, scale_a), (y, scale_b))) - 1`, which is the classical
/// Poincaré upper half-space isometric invariant.
pub fn hyperbolic_distances(
    distances: &ArrayView2<f64>,
    k: usize,
) -> Result<Array2<f64>, ENNError> {
    metric_distances(
        distances,
        k,
        baux::semantics::neighbor::DistanceMetric::Hyperbolic { local_k: k },
        "hyperbolic",
    )
}

/// Local-scale transport dissimilarity matrix.
///
/// Combines geometric-mean scale-damped squared distance `sq_dist /
/// sqrt(scale_a * scale_b)` with 1D radial scale transport `(scale_a -
/// scale_b)^2`. Motivated by isotropic Gaussian `W_2^2` radial transport in
/// Givens and Shortt (*Michigan Math. J.* 31(2):231-240, 1984) and Gelbrich
/// (*Math. Nachr.* 147:185-203, 1990). Dividing `sq_dist` by `sqrt(scale_a *
/// scale_b)` makes it a non-metric dissimilarity rather than exact `W_2^2` or
/// entropic optimal transport.
pub fn wasserstein_distances(
    distances: &ArrayView2<f64>,
    k: usize,
) -> Result<Array2<f64>, ENNError> {
    metric_distances(
        distances,
        k,
        baux::semantics::neighbor::DistanceMetric::Wasserstein { local_k: k },
        "wasserstein",
    )
}

/// Validates subsample log-likelihood inputs.
fn subsample_inputs(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    p: usize,
    num_params: usize,
) -> Result<(), ENNError> {
    if x.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), x.ncols()],
            got: x.shape().to_vec(),
        });
    }
    if y.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![y.nrows(), y.ncols()],
            got: y.shape().to_vec(),
        });
    }
    if x.nrows() != y.nrows() {
        return Err(ENNError::InvalidParameter(format!(
            "x and y must have same number of rows: {} vs {}",
            x.nrows(),
            y.nrows()
        )));
    }
    if p == 0 {
        return Err(ENNError::InvalidParameter(
            "P (num_samples) must be > 0".to_string(),
        ));
    }
    if num_params == 0 {
        return Err(ENNError::InvalidParameter(
            "paramss must be non-empty".to_string(),
        ));
    }
    Ok(())
}

/// Compute single log-likelihood given scaled predictions.
fn compute_loglik(
    y_scaled: &ArrayView2<f64>,
    mu_i: &ArrayView2<f64>,
    se_i: &ArrayView2<f64>,
) -> f64 {
    // Check for non-finite values
    if !mu_i.iter().all(|v| v.is_finite()) || !se_i.iter().all(|v| v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    if se_i.iter().any(|&v| v <= 0.0) {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    for i in 0..y_scaled.nrows() {
        for j in 0..y_scaled.ncols() {
            let y_ij = y_scaled[[i, j]];
            let mu_ij = mu_i[[i, j]];
            let se_ij = se_i[[i, j]];
            let residual = (y_ij - mu_ij) / se_ij;
            loglik +=
                -0.5 * (2.0 * std::f64::consts::PI).ln() - se_ij.ln() - 0.5 * residual * residual;
        }
    }

    if loglik.is_finite() {
        loglik
    } else {
        f64::NEG_INFINITY
    }
}

/// Pure `O(p)` uniform subset sampling without replacement driven by
/// `Key::fold`.
pub fn sampled_key(n: usize, p: usize, key: Key) -> Vec<usize> {
    if p == 0 || n == 0 {
        return Vec::new();
    }
    if p >= n {
        return (0..n).collect();
    }
    let mut out = Vec::with_capacity(p);
    let mut seen = HashSet::with_capacity(p);
    for j in (n - p)..n {
        let word = key.fold(j as u64).seed();
        let pick = ((u128::from(word) * (j as u128 + 1)) >> 64) as usize;
        let chosen = if seen.insert(pick) {
            pick
        } else {
            seen.insert(j);
            j
        };
        out.push(chosen);
    }
    let perm_key = key.fold(0x517C_C1B7_2722_0A95);
    for i in (1..out.len()).rev() {
        let word = perm_key.fold(i as u64).seed();
        let swap = ((u128::from(word) * (i as u128 + 1)) >> 64) as usize;
        out.swap(i, swap);
    }
    out
}

/// Pure leave-one-out likelihood subsampling over model rows driven by `Key`.
pub fn model_loglik(
    model: &ENN,
    paramss: &[ENNParams],
    p: usize,
    key: Key,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = model.len();
    if p == 0 || paramss.is_empty() {
        return Err(ENNError::InvalidParameter(
            "nonempty parameters and p > 0 required".into(),
        ));
    }
    if n <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let indices = sampled_key(n, p, key);
    score_rows(model, &indices, paramss, y_std)
}

/// Subsample model rows for leave-one-out likelihood without full-dataset
/// views. Numerically invalid predictions receive negative infinity, never a
/// valid score.
pub fn subsample_model<R: RngCore + ?Sized>(
    model: &ENN,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let key = Key::from_pair(rng.next_u64(), rng.next_u64());
    model_loglik(model, paramss, p, key, y_std)
}

/// Pure subsample leave-one-out likelihood over complete ordered training data.
pub fn sample_loglik(
    model: &ENN,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    paramss: &[ENNParams],
    p: usize,
    key: Key,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    subsample_inputs(x, y, p, paramss.len())?;

    if x.nrows() != model.len() || x.ncols() != model.num_dim() || y.ncols() != model.num_metrics()
    {
        return Err(ENNError::InvalidParameter(
            "x and y must be complete ordered training rows; use row_loglik for subsets".into(),
        ));
    }
    let n = x.nrows();
    if n == 0 || model.num_obs() <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }

    if !y.iter().all(|v| v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "training targets must be finite".into(),
        ));
    }

    let indices = sampled_key(n, p, key);

    let (stored_x, stored_y, _) = model.natural_rows(&indices)?;
    for (i, &id) in indices.iter().enumerate() {
        if x.row(id) != stored_x.row(i)
            || y.row(id)
                .iter()
                .zip(stored_y.row(i))
                .any(|(&a, &b)| (a - b).abs() > 16.0 * f64::EPSILON * (1.0 + b.abs()))
        {
            return Err(ENNError::InvalidParameter(
                "sampled rows do not match model row order; use row_loglik with explicit IDs"
                    .into(),
            ));
        }
    }
    let scale = y_std
        .map(|s| s.to_owned())
        .unwrap_or_else(|| y.std_axis(Axis(0), 0.0));
    score_rows(model, &indices, paramss, Some(&scale.view()))
}

/// Score a subsample of the complete training data in model row order.
/// Subsets and reordered observations must use `row_loglik` with explicit IDs.
pub fn subsample_loglik<R: RngCore + ?Sized>(
    model: &ENN,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let key = Key::from_pair(rng.next_u64(), rng.next_u64());
    sample_loglik(model, x, y, paramss, p, key, y_std)
}

/// Pure leave-one-out likelihood for explicit model row IDs driven by `Key`.
pub fn rows_loglik(
    model: &ENN,
    rows: &[usize],
    paramss: &[ENNParams],
    p: usize,
    key: Key,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    if p == 0 || paramss.is_empty() || rows.iter().any(|&id| id >= model.len()) {
        return Err(ENNError::InvalidParameter(
            "valid row IDs, nonempty parameters and p > 0 required".into(),
        ));
    }
    let indices = if p >= rows.len() {
        rows.to_vec()
    } else {
        sampled_key(rows.len(), p, key)
            .into_iter()
            .map(|i| rows[i])
            .collect()
    };
    score_rows(model, &indices, paramss, y_std)
}

/// Leave-one-out likelihood for explicit model row IDs, including duplicates.
pub fn row_loglik<R: RngCore + ?Sized>(
    model: &ENN,
    rows: &[usize],
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let key = Key::from_pair(rng.next_u64(), rng.next_u64());
    rows_loglik(model, rows, paramss, p, key, y_std)
}

/// Leave-one-out likelihood from an exact pairwise squared-distance matrix.
///
/// This is equivalent to `row_loglik` for backends that retain observation
/// geometry but cannot afford to materialize every high-dimensional row.
fn validate_loglik(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    p: usize,
) -> Result<(), ENNError> {
    let n = y.nrows();
    if distances.nrows() != n || distances.ncols() != n {
        return Err(ENNError::InvalidShape {
            expected: vec![n, n],
            got: distances.shape().to_vec(),
        });
    }
    if p == 0 || paramss.is_empty() {
        return Err(ENNError::InvalidParameter(
            "nonempty parameters and p > 0 required".into(),
        ));
    }
    if y.ncols() == 0
        || y.iter().any(|value| !value.is_finite())
        || distances
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ENNError::InvalidParameter(
            "distance fitting requires finite outcomes and finite nonnegative distances".into(),
        ));
    }
    if yvar.is_some_and(|values| {
        values.raw_dim() != y.raw_dim()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
    }) {
        return Err(ENNError::InvalidParameter(
            "distance fitting variances must match outcomes and be finite and nonnegative".into(),
        ));
    }
    for params in paramss {
        ENNParams::new(
            params.k_neighbors,
            params.epistemic_scale,
            params.aleatoric_scale,
        )
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    }
    Ok(())
}

fn distance_neighbors(
    distances: &ArrayView2<f64>,
    selected: &[usize],
    max_k: usize,
) -> Vec<Vec<(f64, u32)>> {
    selected
        .iter()
        .map(|&held| {
            let mut row = (0..distances.nrows())
                .filter(|&candidate| candidate != held)
                .map(|candidate| (distances[[held, candidate]], candidate as u32))
                .collect::<Vec<_>>();
            row.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
            row.truncate(max_k);
            row
        })
        .collect()
}

fn output_scale(y: &ArrayView2<f64>, y_std: Option<&ArrayView1<f64>>) -> Array1<f64> {
    y_std
        .map(|values| values.to_owned())
        .unwrap_or_else(|| y.std_axis(Axis(0), 0.0))
        .mapv(|value| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                1.0
            }
        })
}

struct DistanceScore<'a> {
    y_cols: Vec<Vec<f64>>,
    yvar_cols: Option<Vec<Vec<f64>>>,
    selected: &'a [usize],
    neighbors: &'a [Vec<(f64, u32)>],
    scale: &'a Array1<f64>,
    max_k: usize,
}

impl DistanceScore<'_> {
    fn score(&self, params: &ENNParams) -> Result<f64, ENNError> {
        use baux::semantics::neighbor::{UncertaintyLaw, query::posterior_topk};
        use baux::semantics::optimizer::surrogate::loo_point;

        let used = (params.k_neighbors as usize).min(self.max_k);
        let mut total = 0.0;
        for (row, &held) in self.selected.iter().enumerate() {
            let topk = &self.neighbors[row][..used];
            for (metric, outcomes) in self.y_cols.iter().enumerate() {
                let vars = self.yvar_cols.as_ref().map(|cols| cols[metric].as_slice());
                let post = posterior_topk::<{ UncertaintyLaw::InversePrecision }, f64>(
                    topk,
                    outcomes,
                    vars,
                    params.epistemic_scale,
                    params.aleatoric_scale,
                    self.scale[metric],
                    false,
                );
                let ll = loo_point(post.mu, post.se_total, outcomes[held], self.scale[metric]);
                if !ll.is_finite() {
                    return Ok(f64::NEG_INFINITY);
                }
                total += ll;
            }
        }
        Ok(if total.is_finite() {
            total
        } else {
            f64::NEG_INFINITY
        })
    }
}

pub(crate) fn dist_rows(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    selected: &[usize],
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = y.nrows();
    validate_loglik(distances, y, yvar, paramss, selected.len().max(1))?;
    if n <= 1 || selected.is_empty() {
        return Ok(vec![0.0; paramss.len()]);
    }
    let max_k = paramss
        .iter()
        .map(|params| params.k_neighbors as usize)
        .max()
        .unwrap()
        .min(n - 1);
    let neighbors = distance_neighbors(distances, selected, max_k);
    let scale = output_scale(y, y_std);
    if scale.len() != y.ncols() {
        return Err(ENNError::InvalidParameter(
            "y_std must match the number of outputs".into(),
        ));
    }
    let y_cols: Vec<Vec<f64>> = (0..y.ncols())
        .map(|m| (0..n).map(|r| y[[r, m]]).collect())
        .collect();
    let yvar_cols: Option<Vec<Vec<f64>>> = yvar.map(|yv| {
        (0..yv.ncols())
            .map(|m| (0..n).map(|r| yv[[r, m]]).collect())
            .collect()
    });
    let scorer = DistanceScore {
        y_cols,
        yvar_cols,
        selected,
        neighbors: &neighbors,
        scale: &scale,
        max_k,
    };
    paramss.iter().map(|params| scorer.score(params)).collect()
}

/// Pure leave-one-out likelihood from an exact pairwise squared-distance
/// matrix.
pub fn dist_loglik(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    p: usize,
    key: Key,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = y.nrows();
    validate_loglik(distances, y, yvar, paramss, p)?;
    if n <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let selected = sampled_key(n, p, key);
    dist_rows(distances, y, yvar, paramss, &selected, y_std)
}

pub fn distance_loglik<R: RngCore + ?Sized>(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = y.nrows();
    validate_loglik(distances, y, yvar, paramss, p)?;
    if n <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let selected = baux::semantics::neighbor::sampled_rows(n, p, rng);
    dist_rows(distances, y, yvar, paramss, &selected, y_std)
}

fn loo_neighbors(
    model: &ENN,
    rows: &[usize],
    x: &ArrayView2<f64>,
    k: usize,
) -> Result<Vec<Vec<(f64, usize)>>, ENNError> {
    // One bounded lookup for all parameter candidates; never escalate tied rows
    // to a full dataset scan. Exclude identity even if ANN did not return self
    // first.
    let request = i32::try_from(k + 1).map_err(|_| {
        ENNError::InvalidParameter("neighbor request exceeds index capacity".into())
    })?;
    let (distances, ids) = index_search(model, x, request, false, false)?;
    let mut neighbors = Vec::with_capacity(rows.len());
    for (i, &held) in rows.iter().enumerate() {
        let mut row: Vec<(f64, usize)> = ids
            .row(i)
            .iter()
            .zip(distances.row(i))
            .filter_map(|(&id, &d)| {
                (id >= 0 && (id as usize) < model.len() && id as usize != held)
                    .then_some((d, id as usize))
            })
            .collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        row.dedup_by_key(|v| v.1);
        if row.len() < k {
            return Err(ENNError::InvalidParameter(
                "not enough distinct neighbors after held-out exclusion".into(),
            ));
        }
        row.truncate(k);
        neighbors.push(row);
    }
    Ok(neighbors)
}

fn score_rows(
    model: &ENN,
    rows: &[usize],
    paramss: &[ENNParams],
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    validate_params(paramss)?;
    let num_rows = rows.len();
    let n_obs = model.len();
    if num_rows == 0 || n_obs <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let (x, y, _) = model.natural_rows(rows)?;
    let scale = match y_std {
        Some(s) => s.to_owned(),
        None if model.bounded_outputs() => {
            let all_indices: Vec<usize> = (0..n_obs).collect();
            model.natural_rows(&all_indices)?.1.std_axis(Axis(0), 0.0)
        }
        None => model.output_scale().clone(),
    };
    let num_cols = y.ncols();
    if scale.len() != num_cols {
        return Err(ENNError::InvalidParameter(
            "y_std must match the number of outputs".into(),
        ));
    }
    let scale = scale.mapv(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 });
    let y_scaled = &y / &scale;
    let mut k = 0;
    for p in paramss {
        k = k.max(p.k_neighbors as usize);
    }
    k = k.min(n_obs - 1);
    let neighbors = loo_neighbors(model, rows, &x.view(), k)?;
    let mut scores = Vec::with_capacity(paramss.len());
    let mut cache = std::collections::HashMap::new();
    for params in paramss {
        let used = (params.k_neighbors as usize).min(k);
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(used) {
            let dist2s = Array2::from_shape_fn((num_rows, used), |(i, j)| neighbors[i][j].0);
            let idx: Vec<Vec<usize>> = neighbors
                .iter()
                .map(|r| r[..used].iter().map(|v| v.1).collect())
                .collect();
            let mut labels = Array2::zeros((num_rows * used, num_cols));
            for (i, row) in idx.iter().enumerate() {
                for (j, &id) in row.iter().enumerate() {
                    labels
                        .row_mut(i * used + j)
                        .assign(&model.rows().row_y(id)?);
                }
            }
            entry.insert(crate::draw::NeighborData::new(dist2s, idx, labels, used));
        }
        let data = &cache[&used];
        let mut post = compute_posterior(
            model,
            WeightedPosteriorData {
                dist2s: &data.dist2s.view(),
                idx: &data.idx,
                y_neighbors: &data.y_neighbors.view(),
                params,
                observation_noise: true,
                adaptive_neighbors: false,
                yvar_neighbors_override: None,
            },
            None,
        )?;
        if model.bounded_outputs() {
            crate::y_bounds::naturalize(
                &mut post.mu,
                &mut post.se,
                &mut post.se_epi,
                &mut post.se_ale,
                model.y_bounds(),
            );
        }
        scores.push(compute_loglik(
            &y_scaled.view(),
            &(&post.mu / &scale).view(),
            &(&post.se / &scale).view(),
        ));
    }
    Ok(scores)
}

fn validate_params(paramss: &[ENNParams]) -> Result<(), ENNError> {
    for params in paramss {
        ENNParams::new(
            params.k_neighbors,
            params.epistemic_scale,
            params.aleatoric_scale,
        )
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::test_helpers::test_model as create_test_model;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn conformal_geometry() {
        let distances = array![[0.0, 1.0, 100.0], [1.0, 0.0, 81.0], [100.0, 81.0, 0.0]];
        let tuned = conformal_distances(&distances.view(), 1).unwrap();
        assert_eq!(tuned[[0, 1]], 1.0);
        assert!((tuned[[0, 2]] - 100.0 / 9.0).abs() < 1.0e-12);
        assert_eq!(tuned[[0, 2]], tuned[[2, 0]]);
        assert!(tuned.iter().all(|value| value.is_finite()));

        let duplicates = array![[0.0, 0.0], [0.0, 0.0]];
        let tuned = conformal_distances(&duplicates.view(), 1).unwrap();
        assert!(tuned.iter().all(|value| value.is_finite()));

        let ot = wasserstein_distances(&distances.view(), 1).unwrap();
        assert_eq!(ot[[0, 2]], ot[[2, 0]]);
        assert!(ot[[0, 2]] > ot[[0, 1]]);
        assert!(ot.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_001() {
        let model = create_test_model();
        let (x, y, _) = model
            .natural_rows(&(0..model.len()).collect::<Vec<_>>())
            .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
    }

    #[test]
    fn likelihood_rows() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, -0.5]];
        let y = array![[0.2], [-0.4], [1.1], [0.7]];
        let yvar = array![[0.01], [0.04], [0.02], [0.03]];
        let model = ENN::new(
            x.clone(),
            y.clone(),
            Some(yvar.clone()),
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let distances = Array2::from_shape_fn((x.nrows(), x.nrows()), |(left, right)| {
            x.row(left)
                .iter()
                .zip(x.row(right))
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        });
        let paramss = [
            ENNParams::new(2, 0.3, 0.05).unwrap(),
            ENNParams::new(3, 1.7, 0.2).unwrap(),
        ];
        let rows = (0..x.nrows()).collect::<Vec<_>>();
        let scale = y.std_axis(Axis(0), 0.0);
        let mut row_rng = StdRng::seed_from_u64(9);
        let mut distance_rng = StdRng::seed_from_u64(9);
        let expected = row_loglik(
            &model,
            &rows,
            &paramss,
            rows.len(),
            &mut row_rng,
            Some(&scale.view()),
        )
        .unwrap();
        let actual = distance_loglik(
            &distances.view(),
            &y.view(),
            Some(&yvar.view()),
            &paramss,
            rows.len(),
            &mut distance_rng,
            Some(&scale.view()),
        )
        .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn test_002() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(99);
        let via_model = subsample_model(&model, &paramss, 2, &mut rng, None).unwrap();
        let mut rng2 = StdRng::seed_from_u64(99);
        let all: Vec<usize> = (0..model.len()).collect();
        let (full_x, full_y, _) = model.rows().train_rows(&all).unwrap();
        let via_views = subsample_loglik(
            &model,
            &full_x.view(),
            &full_y.view(),
            &paramss,
            2,
            &mut rng2,
            None,
        )
        .unwrap();
        assert_eq!(via_model.len(), via_views.len());
        assert!(via_model[0].is_finite());
    }

    #[test]
    fn test_003() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks, vec![0.0]);
    }

    #[test]
    fn test_008() {
        let model = create_test_model();
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 0, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_009() {
        let model = create_test_model();
        let x = array![[0.5, 0.5], [0.2, 0.8]]; // 2 rows
        let y = array![[1.0]]; // 1 row
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_010() {
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let err = subsample_inputs(&x.view(), &y.view(), 1, 0).unwrap_err();
        assert!(err.to_string().contains("paramss must be non-empty"));
    }

    #[test]
    fn test_011() {
        let y_scaled = array![[1.0]];
        let mu_i = array![[0.0]];
        let se_i = array![[1.0]];
        let ll = compute_loglik(&y_scaled.view(), &mu_i.view(), &se_i.view());
        let expected = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn test_012() {
        let model = create_test_model();
        let (x, y, _) = model
            .natural_rows(&(0..model.len()).collect::<Vec<_>>())
            .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let y_std = array![2.0];
        let mut rng = StdRng::seed_from_u64(11);
        let paramss = vec![params];
        let logliks = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut rng,
            Some(&y_std.view()),
        )
        .unwrap();
        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
        let again = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut StdRng::seed_from_u64(11),
            Some(&y_std.view()),
        )
        .unwrap();
        assert!((logliks[0] - again[0]).abs() < 1e-12);
    }

    #[test]
    fn test_013() {
        let y = array![[1.0]];
        let mu_bad = array![[f64::NAN]];
        let se_ok = array![[1.0]];
        assert_eq!(
            compute_loglik(&y.view(), &mu_bad.view(), &se_ok.view()),
            f64::NEG_INFINITY
        );

        let mu_ok = array![[1.0]];
        let se_bad = array![[0.0]];
        assert_eq!(
            compute_loglik(&y.view(), &mu_ok.view(), &se_bad.view()),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn likelihood_limits() {
        let value = array![[0.0]];
        for se in [1e-200, 1e200] {
            let actual = compute_loglik(&value.view(), &value.view(), &array![[se]].view());
            let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - se.ln();
            assert!(actual.is_finite());
            assert_eq!(actual, expected);
        }
        for se in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(
                compute_loglik(&value.view(), &value.view(), &array![[se]].view()),
                f64::NEG_INFINITY
            );
        }
    }
}
