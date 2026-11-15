//! Posterior draw computation helpers.

use ndarray::Array3;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::draw::DrawInternals;
use crate::error::ENNError;
use crate::hash::{normal_metric, unique_inverse};
use crate::model::ENN;
#[cfg(not(target_arch = "wasm32"))]
use baux::neighbor_index::parallel::use_rayon;

struct DrawMultiView<'a> {
    unique_indices: &'a [i64],
    inverse: &'a [usize],
    scale: &'a [f64],
    w_flat: &'a [f64],
    mu_flat: &'a [f64],
    n: usize,
    k: usize,
    m: usize,
}

fn fill_draw_seed(
    view: &DrawMultiView<'_>,
    seed: i64,
    unique_cache: &mut [f64],
    out_seed: &mut [f64],
) {
    let seed_u64 = seed as u64;
    let m = view.m;
    let k = view.k;
    for (ui, &unique_idx) in view.unique_indices.iter().enumerate() {
        for j in 0..m {
            unique_cache[ui * m + j] = normal_metric(seed_u64, unique_idx, j);
        }
    }
    for i in 0..view.n {
        for j in 0..m {
            let mut weighted_u = 0.0;
            for ki in 0..k {
                let inv = view.inverse[i * k + ki];
                let w = view.w_flat[(i * k + ki) * m + j];
                weighted_u += w * unique_cache[inv * m + j];
            }
            out_seed[i * m + j] = view.mu_flat[i * m + j] + view.scale[i * m + j] * weighted_u;
        }
    }
}

fn fill_draw_scalar_seed(
    unique_indices: &[i64],
    inverse: &[usize],
    scale: &[f64],
    w_flat: &[f64],
    mu_flat: &[f64],
    n: usize,
    k: usize,
    seed: i64,
    unique_cache: &mut [f64],
    out_seed: &mut [f64],
) {
    let seed_u64 = seed as u64;
    for (ui, &unique_idx) in unique_indices.iter().enumerate() {
        unique_cache[ui] = normal_metric(seed_u64, unique_idx, 0);
    }
    for i in 0..n {
        let mut weighted_u = 0.0;
        let base = i * k;
        for ki in 0..k {
            weighted_u += w_flat[base + ki] * unique_cache[inverse[base + ki]];
        }
        out_seed[i] = mu_flat[i] + scale[i] * weighted_u;
    }
}

pub(crate) fn draw_internals(
    model: &ENN,
    internals: &DrawInternals,
    function_seeds: &[i64],
) -> Result<Array3<f64>, ENNError> {
    let n = internals.mu.nrows();
    let k = internals.n_neighbors();
    let m = model.num_metrics();
    let num_seeds = function_seeds.len();

    if k == 0 {
        return Ok(draw_empty(internals, num_seeds, n, m));
    }

    // Fuse hash → weighted sum per seed: never materialize u[s,i,k,j].
    let idx_array: Vec<i64> = internals.idx.iter().flatten().map(|&i| i as i64).collect();
    let (unique_indices, inverse) = unique_inverse(&idx_array);
    let n_unique = unique_indices.len();

    if m == 1 {
        return draw_scalar(
            internals,
            function_seeds,
            &unique_indices,
            &inverse,
            n,
            k,
            n_unique,
        );
    }

    let out_stride = n * m;
    let (scale, w_flat, mu_flat) = pack_moments(internals, n, k, m);

    let mut draws = Array3::zeros((num_seeds, n, m));
    let draws_flat = draws
        .as_slice_memory_order_mut()
        .ok_or_else(|| ENNError::InvalidParameter("draws must be contiguous".into()))?;

    let view = DrawMultiView {
        unique_indices: &unique_indices,
        inverse: &inverse,
        scale: &scale,
        w_flat: &w_flat,
        mu_flat: &mu_flat,
        n,
        k,
        m,
    };

    #[cfg(not(target_arch = "wasm32"))]
    let ran_parallel = if use_rayon() {
        draws_flat
            .par_chunks_mut(out_stride)
            .zip(function_seeds.par_iter())
            .for_each_init(
                || vec![0.0f64; n_unique * m],
                |unique_cache, (out_seed, &seed)| {
                    fill_draw_seed(&view, seed, unique_cache, out_seed);
                },
            );
        true
    } else {
        false
    };
    #[cfg(target_arch = "wasm32")]
    let ran_parallel = false;

    if !ran_parallel {
        let mut unique_cache = vec![0.0f64; n_unique * m];
        for (seed_offset, out_seed) in draws_flat.chunks_mut(out_stride).enumerate() {
            fill_draw_seed(
                &view,
                function_seeds[seed_offset],
                &mut unique_cache,
                out_seed,
            );
        }
    }

    Ok(draws)
}

/// Specialized fused draw path for the common single-metric case.
fn draw_scalar(
    internals: &DrawInternals,
    function_seeds: &[i64],
    unique_indices: &[i64],
    inverse: &[usize],
    n: usize,
    k: usize,
    n_unique: usize,
) -> Result<Array3<f64>, ENNError> {
    let num_seeds = function_seeds.len();
    let mut scale = vec![0.0f64; n];
    let mut mu_flat = vec![0.0f64; n];
    let mut w_flat = vec![0.0f64; n * k];
    for i in 0..n {
        let l2_safe = internals.l2[[i, 0]].max(1e-12);
        scale[i] = internals.se[[i, 0]] / l2_safe;
        mu_flat[i] = internals.mu[[i, 0]];
        for ki in 0..k {
            w_flat[i * k + ki] = internals.w_normalized[[i, ki, 0]];
        }
    }

    let mut draws = Array3::zeros((num_seeds, n, 1));
    let draws_flat = draws
        .as_slice_memory_order_mut()
        .ok_or_else(|| ENNError::InvalidParameter("draws must be contiguous".into()))?;

    #[cfg(not(target_arch = "wasm32"))]
    let ran_parallel = if use_rayon() {
        draws_flat
            .par_chunks_mut(n)
            .zip(function_seeds.par_iter())
            .for_each_init(
                || vec![0.0f64; n_unique],
                |unique_cache, (out_seed, &seed)| {
                    fill_draw_scalar_seed(
                        unique_indices,
                        inverse,
                        &scale,
                        &w_flat,
                        &mu_flat,
                        n,
                        k,
                        seed,
                        unique_cache,
                        out_seed,
                    );
                },
            );
        true
    } else {
        false
    };
    #[cfg(target_arch = "wasm32")]
    let ran_parallel = false;

    if !ran_parallel {
        let mut unique_cache = vec![0.0f64; n_unique];
        for (seed_offset, out_seed) in draws_flat.chunks_mut(n).enumerate() {
            fill_draw_scalar_seed(
                unique_indices,
                inverse,
                &scale,
                &w_flat,
                &mu_flat,
                n,
                k,
                function_seeds[seed_offset],
                &mut unique_cache,
                out_seed,
            );
        }
    }

    Ok(draws)
}

fn draw_empty(internals: &DrawInternals, num_seeds: usize, n: usize, m: usize) -> Array3<f64> {
    let mut draws = Array3::zeros((num_seeds, n, m));
    for s in 0..num_seeds {
        for i in 0..n {
            for j in 0..m {
                draws[[s, i, j]] = internals.mu[[i, j]];
            }
        }
    }
    draws
}

fn pack_moments(
    internals: &DrawInternals,
    n: usize,
    k: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut scale = vec![0.0f64; n * m];
    let mut w_flat = vec![0.0f64; n * k * m];
    let mut mu_flat = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let l2_safe = internals.l2[[i, j]].max(1e-12);
            scale[i * m + j] = internals.se[[i, j]] / l2_safe;
            mu_flat[i * m + j] = internals.mu[[i, j]];
            for ki in 0..k {
                w_flat[(i * k + ki) * m + j] = internals.w_normalized[[i, ki, j]];
            }
        }
    }

    (scale, w_flat, mu_flat)
}
