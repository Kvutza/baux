//! `#![no_std]` multi-objective (MORBO) Chebyshev scalarization, simplex weight
//! sampling, non-dominated sorting (NDS), and exact 2D hypervolume primitives
//! parameterized over generic [`Real`] scalars (`f32` and `f64`).

use super::key::Key;
use super::real::Real;

const TAG_SIMPLEX: u64 = 0x4d4f_5242_4f5f_5754;
const TAG_PARETO: u64 = 0x5041_5245_544f_524b;

/// Rescalarization cadence shared across Host, Metal, and CUDA MORBO engines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RescalarizeMode {
    #[default]
    OnRestart,
    OnPropose,
}

/// Deterministically sample `out.len()` positive simplex weights summing to
/// `1` from `Dirichlet(1, ..., 1)` using pure [`Key::fold`] transitions.
#[inline]
pub fn sample_simplex<T: Real>(key: Key, out: &mut [T]) {
    if out.is_empty() {
        return;
    }
    let base = key.fold(TAG_SIMPLEX);
    let mut sum = T::ZERO;
    for (m, slot) in out.iter_mut().enumerate() {
        let u = base.fold(m as u64).open::<T>();
        let g = (-u.ln()).max(T::VAR_FLOOR);
        *slot = g;
        sum += g;
    }
    let inv = if sum > T::ZERO && sum.is_finite() {
        T::ONE / sum
    } else {
        T::ONE / T::from_idx(out.len())
    };
    for slot in out.iter_mut() {
        *slot = if sum > T::ZERO && sum.is_finite() {
            *slot * inv
        } else {
            inv
        };
    }
}

/// Compute per-metric finite `(min, max)` ranges over row-major `(rows, cols)`
/// observations `y`. Falls back to `(0, 0)` when a column has no finite
/// entries.
#[inline]
pub fn metric_ranges<T: Real>(
    y: &[T],
    rows: usize,
    cols: usize,
    out_min: &mut [T],
    out_max: &mut [T],
) {
    let m_len = cols.min(out_min.len()).min(out_max.len());
    for m in 0..m_len {
        out_min[m] = T::INFINITY;
        out_max[m] = T::NEG_INFINITY;
    }
    for r in 0..rows {
        let base = r * cols;
        if base + m_len > y.len() {
            break;
        }
        for m in 0..m_len {
            let v = y[base + m];
            if v.is_finite() {
                if v < out_min[m] {
                    out_min[m] = v;
                }
                if v > out_max[m] {
                    out_max[m] = v;
                }
            }
        }
    }
    for m in 0..m_len {
        if !out_min[m].is_finite() || !out_max[m].is_finite() {
            out_min[m] = T::ZERO;
            out_max[m] = T::ZERO;
        }
    }
}

/// Augmented Chebyshev scalarization `min_m (w_m * z_m) + alpha * sum_m (w_m *
/// z_m)` for a single `M`-objective row.
#[inline]
pub fn chebyshev<T: Real>(
    y_row: &[T],
    y_min: &[T],
    y_max: &[T],
    weights: &[T],
    alpha: T,
    clip: bool,
) -> T {
    let m_len = y_row
        .len()
        .min(y_min.len())
        .min(y_max.len())
        .min(weights.len());
    if m_len == 0 {
        return T::NEG_INFINITY;
    }
    let mut t_min = T::INFINITY;
    let mut t_sum = T::ZERO;
    for m in 0..m_len {
        let v = y_row[m];
        if !v.is_finite() {
            return T::NEG_INFINITY;
        }
        let denom = y_max[m] - y_min[m];
        let z = if !denom.is_finite() || denom <= T::ZERO {
            T::HALF
        } else {
            let raw = (v - y_min[m]) / denom;
            if clip {
                raw.clamp(T::ZERO, T::ONE)
            } else {
                raw
            }
        };
        let t = z * weights[m];
        if t < t_min {
            t_min = t;
        }
        t_sum += t;
    }
    t_min + alpha * t_sum
}

/// Scalarize `rows` rows of row-major `(rows, cols)` matrix `y` into `out`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn scalarize_slice<T: Real>(
    y: &[T],
    rows: usize,
    cols: usize,
    y_min: &[T],
    y_max: &[T],
    weights: &[T],
    alpha: T,
    clip: bool,
    out: &mut [T],
) {
    let n = rows.min(out.len());
    for (r, dst) in out.iter_mut().enumerate().take(n) {
        let start = r * cols;
        if start + cols > y.len() {
            *dst = T::NEG_INFINITY;
        } else {
            *dst = chebyshev(&y[start..start + cols], y_min, y_max, weights, alpha, clip);
        }
    }
}

/// Pareto maximization dominance: `left` dominates `right` iff `left[m] >=
/// right[m]` for all `m` and `left[m] > right[m]` for at least one `m` (NaN
/// rows never dominate finite rows).
#[inline]
pub fn dominates<T: Real>(left: &[T], right: &[T]) -> bool {
    let cols = left.len().min(right.len());
    if cols == 0 {
        return false;
    }
    let mut strictly_better = false;
    for m in 0..cols {
        let a = left[m];
        let b = right[m];
        if a.is_nan() {
            return false;
        }
        if b.is_nan() {
            strictly_better = true;
            continue;
        }
        let ord = a.total_cmp(&b);
        if ord.is_lt() {
            return false;
        }
        strictly_better |= ord.is_gt();
    }
    strictly_better
}

/// Zero-allocation non-dominated sorting (NDS) writing 0-based Pareto front
/// rank `0, 1, 2, ...` for each row into `out_ranks[..rows]`.
pub fn nds_ranks<T: Real>(objectives: &[T], rows: usize, cols: usize, out_ranks: &mut [u32]) {
    let n = rows.min(out_ranks.len());
    if n == 0 || cols == 0 || objectives.len() < n * cols {
        return;
    }
    for slot in out_ranks.iter_mut().take(n) {
        *slot = u32::MAX;
    }
    let mut assigned = 0usize;
    let mut front = 0u32;
    while assigned < n {
        let pending_tag = u32::MAX - 1;
        let mut progressed = false;
        for i in 0..n {
            if out_ranks[i] != u32::MAX {
                continue;
            }
            let row_i = &objectives[i * cols..(i + 1) * cols];
            let mut dom = false;
            for j in 0..n {
                if i == j || out_ranks[j] < pending_tag {
                    continue;
                }
                let row_j = &objectives[j * cols..(j + 1) * cols];
                if dominates(row_j, row_i) {
                    dom = true;
                    break;
                }
            }
            if !dom {
                out_ranks[i] = pending_tag;
                progressed = true;
            }
        }
        if !progressed {
            for r in out_ranks.iter_mut().take(n) {
                if *r == u32::MAX {
                    *r = front;
                }
            }
            break;
        }
        for r in out_ranks.iter_mut().take(n) {
            if *r == pending_tag {
                *r = front;
                assigned += 1;
            }
        }
        front = front.saturating_add(1);
    }
}

/// Select up to `out_selected.len()` candidates by ascending NDS front rank,
/// breaking ties within the overflow front via deterministic [`Key::fold`]
/// priorities. Returns the number of selected indices written.
pub fn pareto_select<T: Real>(
    objectives: &[T],
    rows: usize,
    cols: usize,
    ranks_scratch: &mut [u32],
    out_selected: &mut [usize],
    key: Key,
) -> usize {
    let n = rows.min(ranks_scratch.len());
    let k = out_selected.len().min(n);
    if k == 0 || cols == 0 {
        return 0;
    }
    nds_ranks(objectives, n, cols, &mut ranks_scratch[..n]);
    let prio_key = key.fold(TAG_PARETO);
    let mut count = 0usize;
    let mut front = 0u32;
    while count < k && (front as usize) <= n {
        let mut front_size = 0usize;
        for &r in ranks_scratch.iter().take(n) {
            if r == front {
                front_size += 1;
            }
        }
        if front_size == 0 {
            front += 1;
            continue;
        }
        if count + front_size <= k {
            for (idx, &r) in ranks_scratch.iter().enumerate().take(n) {
                if r == front {
                    out_selected[count] = idx;
                    count += 1;
                }
            }
        } else {
            let need = k - count;
            for _ in 0..need {
                let mut best_idx = 0usize;
                let mut best_prio = 0u64;
                let mut found = false;
                for (idx, &r) in ranks_scratch.iter().enumerate().take(n) {
                    if r != front {
                        continue;
                    }
                    let p = prio_key.fold(idx as u64).seed();
                    if !found || p > best_prio || (p == best_prio && idx < best_idx) {
                        found = true;
                        best_prio = p;
                        best_idx = idx;
                    }
                }
                if !found {
                    break;
                }
                ranks_scratch[best_idx] = u32::MAX;
                out_selected[count] = best_idx;
                count += 1;
            }
            break;
        }
        front += 1;
    }
    count
}

/// Exact 2D hypervolume over points already filtered (`x0 > ref0 && x1 > ref1`)
/// and sorted by `x0` descending.
#[inline]
pub fn hv_walk<T: Real>(sorted_desc: &[(T, T)], ref0: T, ref1: T) -> T {
    if sorted_desc.is_empty() || !ref0.is_finite() || !ref1.is_finite() {
        return T::ZERO;
    }
    let mut hv = T::ZERO;
    let mut best_y1 = ref1;
    for i in 0..sorted_desc.len() {
        let (x0, y1) = sorted_desc[i];
        if y1 > best_y1 {
            best_y1 = y1;
        }
        let x_next = if i + 1 < sorted_desc.len() {
            sorted_desc[i + 1].0
        } else {
            ref0
        };
        hv += (x0 - x_next) * (best_y1 - ref1);
    }
    hv
}

/// Zero-allocation exact 2D hypervolume on row-major `(rows, 2)` slice `y`
/// using caller scratch `scratch[..rows]`.
pub fn hv_pair<T: Real>(y: &[T], rows: usize, ref0: T, ref1: T, scratch: &mut [(T, T)]) -> T {
    if !ref0.is_finite() || !ref1.is_finite() {
        return T::ZERO;
    }
    let n = rows.min(scratch.len()).min(y.len() / 2);
    let mut m = 0usize;
    for r in 0..n {
        let x0 = y[2 * r];
        let x1 = y[2 * r + 1];
        if x0.is_finite() && x1.is_finite() && x0 > ref0 && x1 > ref1 {
            scratch[m] = (x0, x1);
            m += 1;
        }
    }
    scratch[..m].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    hv_walk(&scratch[..m], ref0, ref1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_chebyshev() {
        let mut w = [0.0_f64; 3];
        sample_simplex(Key::new(2026), &mut w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w.iter().all(|&x| x > 0.0));

        let mut w32 = [0.0_f32; 3];
        sample_simplex(Key::new(2026), &mut w32);
        assert!((w32.iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let y = [1.0, 2.0, 1.0, 3.0];
        let mut out = [0.0; 2];
        scalarize_slice(
            &y,
            2,
            2,
            &[1.0, 2.0],
            &[1.0, 5.0],
            &[0.5, 0.5],
            0.1,
            true,
            &mut out,
        );
        assert!((out[0] - 0.025).abs() < 1e-12);
        assert!((out[1] - 0.208_333_333_333_333_34).abs() < 1e-12);
    }

    #[test]
    fn nds_hv() {
        let objs = [3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.5, 0.5];
        let mut ranks = [0u32; 4];
        nds_ranks(&objs, 4, 2, &mut ranks);
        assert_eq!(ranks, [0, 0, 0, 1]);

        let mut sel = [0usize; 2];
        let n = pareto_select(&objs, 4, 2, &mut ranks, &mut sel, Key::new(42));
        assert_eq!(n, 2);
        assert!(sel.iter().all(|&i| i < 3));

        let pts = [1.0, 0.5, 0.5, 1.0, 0.2, 0.2];
        let mut scratch = [(0.0, 0.0); 3];
        let hv = hv_pair(&pts, 3, 0.0, 0.0, &mut scratch);
        assert!((hv - 0.75).abs() < 1e-12);
    }
}
