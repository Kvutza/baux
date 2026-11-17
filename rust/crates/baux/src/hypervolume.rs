//! Exact hypervolume calculation for bounded maximization problems.

use ndarray::{ArrayView1, ArrayView2};
use thiserror::Error;

/// Errors that can occur during hypervolume calculation.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HypervolumeError {
    /// Input array has wrong dimensionality.
    #[error("y must be 2-dimensional, got shape {0:?}")]
    InvalidDimension(Vec<usize>),
    /// Input array has wrong number of columns.
    #[error("y must have 2 columns, got shape {0:?}")]
    InvalidColumnCount(Vec<usize>),
    /// Reference point has wrong shape.
    #[error("ref_point must have 2 elements, got shape {0:?}")]
    InvalidRefPoint(usize),
    /// Input and reference dimensions disagree.
    #[error("y has {columns} columns but ref_point has {reference} elements")]
    DimensionMismatch { columns: usize, reference: usize },
}

/// Exact dominated hypervolume for maximization in one or more dimensions.
///
/// Dominated rows are pruned once and recursive slicing operates over flat
/// index views without cloning point vectors at each axis level.
pub fn hypervolume_max(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    if y.ndim() != 2 {
        return Err(HypervolumeError::InvalidDimension(y.shape().to_vec()));
    }
    if y.ncols() != ref_point.len() {
        return Err(HypervolumeError::DimensionMismatch {
            columns: y.ncols(),
            reference: ref_point.len(),
        });
    }
    if y.ncols() == 0 || y.nrows() == 0 || !ref_point.iter().all(|v| v.is_finite()) {
        return Ok(0.0);
    }
    if y.ncols() == 2 {
        return hypervolume2d_max(y, ref_point);
    }
    let m = y.ncols();
    let mut flat = Vec::with_capacity(y.nrows() * m);
    let mut count = 0usize;
    for row in y.rows() {
        if row
            .iter()
            .zip(ref_point.iter())
            .all(|(val, r)| val.is_finite() && val > r)
        {
            flat.extend(row.iter().copied());
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    let active = pareto_indices(&flat, count, m);
    let reference: Vec<f64> = ref_point.iter().copied().collect();
    Ok(slice_flat(&flat, &active, m, m, &reference))
}

/// Pure flat-slice exact hypervolume over row-major `(rows, cols)` values.
pub fn hypervolume_slice(
    y: &[f64],
    rows: usize,
    cols: usize,
    ref_point: &[f64],
) -> Result<f64, HypervolumeError> {
    if cols != ref_point.len() {
        return Err(HypervolumeError::DimensionMismatch {
            columns: cols,
            reference: ref_point.len(),
        });
    }
    if y.len() != rows.checked_mul(cols).unwrap_or(usize::MAX) {
        return Err(HypervolumeError::InvalidDimension(vec![y.len()]));
    }
    if cols == 0 || rows == 0 || !ref_point.iter().all(|v| v.is_finite()) {
        return Ok(0.0);
    }
    let view = ArrayView2::from_shape((rows, cols), y)
        .map_err(|_| HypervolumeError::InvalidDimension(vec![rows, cols]))?;
    let rview = ArrayView1::from(ref_point);
    hypervolume_max(&view, &rview)
}

/// Exact exclusive hypervolume contribution `HV(Y) - HV(Y \ {y_i})` per row.
pub fn hv_contributions(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<Vec<f64>, HypervolumeError> {
    if y.ndim() != 2 {
        return Err(HypervolumeError::InvalidDimension(y.shape().to_vec()));
    }
    if y.ncols() != ref_point.len() {
        return Err(HypervolumeError::DimensionMismatch {
            columns: y.ncols(),
            reference: ref_point.len(),
        });
    }
    let nrows = y.nrows();
    let m = y.ncols();
    if m == 0 || nrows == 0 || !ref_point.iter().all(|v| v.is_finite()) {
        return Ok(vec![0.0; nrows]);
    }
    let reference: Vec<f64> = ref_point.iter().copied().collect();
    let mut flat = Vec::with_capacity(nrows * m);
    let mut orig_rows = Vec::with_capacity(nrows);
    for (r, row) in y.rows().into_iter().enumerate() {
        if row
            .iter()
            .zip(&reference)
            .all(|(val, rf)| val.is_finite() && val > rf)
        {
            flat.extend(row.iter().copied());
            orig_rows.push(r);
        }
    }
    let count = orig_rows.len();
    let mut out = vec![0.0; nrows];
    if count == 0 {
        return Ok(out);
    }
    let frontier = pareto_indices(&flat, count, m);
    let total = slice_flat(&flat, &frontier, m, m, &reference);
    let mut subset = Vec::with_capacity(frontier.len().saturating_sub(1));
    for &idx in &frontier {
        let row_slice = &flat[idx * m..(idx + 1) * m];
        let has_duplicate =
            (0..count).any(|other| other != idx && flat[other * m..(other + 1) * m] == *row_slice);
        if has_duplicate {
            continue;
        }
        subset.clear();
        subset.extend(frontier.iter().copied().filter(|&other| other != idx));
        let without = slice_flat(&flat, &subset, m, m, &reference);
        out[orig_rows[idx]] = (total - without).max(0.0);
    }
    Ok(out)
}

/// Noise-aware hypervolume evaluating `mu - z * sqrt(yvar)` against
/// `ref_point`.
pub fn noisy_hv(
    mu: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    ref_point: &ArrayView1<f64>,
    z: f64,
) -> Result<f64, HypervolumeError> {
    let Some(yv) = yvar else {
        return hypervolume_max(mu, ref_point);
    };
    if mu.shape() != yv.shape() {
        return Err(HypervolumeError::InvalidDimension(yv.shape().to_vec()));
    }
    let adjusted = ndarray::Array2::from_shape_fn(mu.raw_dim(), |(i, j)| {
        let var = yv[[i, j]];
        if var.is_finite() && var >= 0.0 {
            mu[[i, j]] - z * var.sqrt()
        } else {
            f64::NAN
        }
    });
    hypervolume_max(&adjusted.view(), ref_point)
}

#[inline]
fn dominates_row(left: &[f64], right: &[f64]) -> bool {
    baux::semantics::morbo::dominates(left, right)
}

fn pareto_indices(flat: &[f64], count: usize, m: usize) -> Vec<usize> {
    (0..count)
        .filter(|&i| {
            let row_i = &flat[i * m..(i + 1) * m];
            !(0..count).any(|other| {
                if other == i {
                    return false;
                }
                let row_o = &flat[other * m..(other + 1) * m];
                dominates_row(row_o, row_i) || (row_o == row_i && other < i)
            })
        })
        .collect()
}

fn frontier_area(flat: &[f64], active: &[usize], m: usize, ref0: f64, ref1: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = active
        .iter()
        .map(|&idx| (flat[idx * m], flat[idx * m + 1]))
        .collect();
    pts.sort_by(|a, b| b.0.total_cmp(&a.0));
    baux::semantics::morbo::hv_walk(&pts, ref0, ref1)
}

fn slice_flat(flat: &[f64], active: &[usize], m: usize, dim: usize, reference: &[f64]) -> f64 {
    if active.is_empty() {
        return 0.0;
    }
    if dim == 1 {
        return active
            .iter()
            .map(|&idx| flat[idx * m])
            .fold(reference[0], f64::max)
            - reference[0];
    }
    if dim == 2 {
        return frontier_area(flat, active, m, reference[0], reference[1]);
    }
    let axis = dim - 1;
    let mut edges: Vec<f64> = active.iter().map(|&idx| flat[idx * m + axis]).collect();
    edges.sort_by(f64::total_cmp);
    edges.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let mut previous = reference[axis];
    let mut volume = 0.0;
    let mut projected = Vec::with_capacity(active.len());
    for edge in edges {
        projected.clear();
        projected.extend(
            active
                .iter()
                .copied()
                .filter(|&idx| flat[idx * m + axis] >= edge),
        );
        volume += (edge - previous) * slice_flat(flat, &projected, m, axis, &reference[..axis]);
        previous = edge;
    }
    volume
}

/// Calculate 2D hypervolume for maximization problems.
///
/// This implements the "walking frontier" algorithm for computing
/// the hypervolume indicator in 2D objective space.
///
/// # Arguments
///
/// * `y` - Array of shape (N, 2) containing objective values to maximize
/// * `ref_point` - Array of shape (2,) containing reference point for
///   hypervolume
///
/// # Returns
///
/// The hypervolume as a float, or an error if inputs are invalid.
///
/// # Algorithm
///
/// 1. Filter points that dominate the reference point in both dimensions
/// 2. Sort remaining points by first objective in descending order (stable)
/// 3. Walk the frontier, accumulating rectangular areas
///
/// # Example
///
/// ```
/// use baux::hypervolume2d_max;
/// use ndarray::array;
///
/// let y = array![[1.0, 0.5], [0.5, 1.0]];
/// let ref_point = array![0.0, 0.0];
/// let hv = hypervolume2d_max(&y.view(), &ref_point.view()).unwrap();
/// assert!((hv - 0.75).abs() < 1e-10);
/// ```
pub fn hypervolume2d_max(
    y: &ArrayView2<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    // Validate dimensions
    if y.ndim() != 2 {
        return Err(HypervolumeError::InvalidDimension(y.shape().to_vec()));
    }
    if y.ncols() != 2 {
        return Err(HypervolumeError::InvalidColumnCount(y.shape().to_vec()));
    }
    if ref_point.len() != 2 {
        return Err(HypervolumeError::InvalidRefPoint(ref_point.len()));
    }

    // Handle empty input
    if y.nrows() == 0 {
        return Ok(0.0);
    }

    let ref0 = ref_point[0];
    let ref1 = ref_point[1];
    if !ref0.is_finite() || !ref1.is_finite() {
        return Ok(0.0);
    }

    // Collect points that dominate reference point in both dimensions
    let mut dominating: Vec<(f64, f64)> = Vec::with_capacity(y.nrows());
    for row in y.rows() {
        let x0 = row[0];
        let x1 = row[1];
        if x0.is_finite() && x1.is_finite() && x0 > ref0 && x1 > ref1 {
            dominating.push((x0, x1));
        }
    }

    // If no points dominate, hypervolume is zero
    if dominating.is_empty() {
        return Ok(0.0);
    }

    // Sort by first objective in descending order (stable sort for parity).
    // total_cmp avoids panic if NaNs ever reach this point.
    dominating.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Walking frontier algorithm
    let mut hv = 0.0;
    let mut best_y1 = ref1;

    for i in 0..dominating.len() {
        let (x0, y1) = dominating[i];

        // Update best y1 seen so far
        if y1 > best_y1 {
            best_y1 = y1;
        }

        // Next x coordinate (reference if at end)
        let x_next = if i + 1 < dominating.len() {
            dominating[i + 1].0
        } else {
            ref0
        };

        // Accumulate rectangle area
        hv += (x0 - x_next) * (best_y1 - ref1);
    }

    Ok(hv)
}

/// Convenience function for 1D arrays.
///
/// # Panics
///
/// Panics if the conversion from 1D to 2D view fails.
pub fn hypervolume_1d(
    y: &ArrayView1<f64>,
    ref_point: &ArrayView1<f64>,
) -> Result<f64, HypervolumeError> {
    let y_2d = y.view().insert_axis(ndarray::Axis(0));
    hypervolume2d_max(&y_2d, ref_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayView2, array};

    fn hypervolume_origin(y: &ArrayView2<f64>) -> f64 {
        let ref_point = array![0.0, 0.0];
        hypervolume2d_max(y, &ref_point.view()).unwrap()
    }

    #[test]
    fn test_emptyarray() {
        let y = Array2::<f64>::zeros((0, 2));
        assert_eq!(hypervolume_origin(&y.view()), 0.0);
    }

    #[test]
    fn test_nodominatingpoints() {
        let y = array![[-1.0, -1.0], [-0.5, -0.5]];
        assert_eq!(hypervolume_origin(&y.view()), 0.0);
    }

    #[test]
    fn test_simplehypervolume() {
        let y = array![[1.0, 0.5], [0.5, 1.0]];
        let result = hypervolume_origin(&y.view());
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_threepoints() {
        let y = array![[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]];
        let result = hypervolume_origin(&y.view());
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn volume_3d() {
        let y = array![[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]];
        let reference = array![0.0, 0.0, 0.0];
        let volume = hypervolume_max(&y.view(), &reference.view()).unwrap();
        assert!((volume - 1.25).abs() < 1e-12);
    }

    #[test]
    fn volume_parity() {
        let y = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.2]];
        let reference = array![0.0, 0.0];
        let exact = hypervolume_max(&y.view(), &reference.view()).unwrap();
        let specialized = hypervolume2d_max(&y.view(), &reference.view()).unwrap();
        assert!((exact - specialized).abs() < 1e-12);
    }

    #[test]
    fn test_withnandoesnotpanic() {
        let y = array![[1.0, 0.5], [f64::NAN, 2.0], [0.8, 0.3]];
        let ref_point = array![0.0, 0.0];
        let result = hypervolume2d_max(&y.view(), &ref_point.view()).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_invalidcolumncount() {
        let y = array![[1.0, 0.5, 0.3]]; // 3 columns instead of 2
        let ref_point = array![0.0, 0.0];
        let result = hypervolume2d_max(&y.view(), &ref_point.view());
        assert!(matches!(
            result,
            Err(HypervolumeError::InvalidColumnCount(_))
        ));
    }

    #[test]
    fn test_invalidrefpoint() {
        let y = array![[1.0, 0.5]];
        let ref_point = array![0.0]; // Wrong length
        let result = hypervolume2d_max(&y.view(), &ref_point.view());
        assert!(matches!(result, Err(HypervolumeError::InvalidRefPoint(1))));
    }

    #[test]
    fn test_hypervolume2dmax1d() {
        let y = array![1.0, 0.5];
        let ref_point = array![0.0, 0.0];
        let result = hypervolume_1d(&y.view(), &ref_point.view()).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn contributions_noisy() {
        let y = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.2], [1.0, 0.5]];
        let reference = array![0.0, 0.0];
        let contrib = hv_contributions(&y.view(), &reference.view()).unwrap();
        assert_eq!(contrib.len(), 4);
        assert_eq!(contrib[0], 0.0); // duplicate with row 3 -> 0 exclusive
        assert!((contrib[1] - 0.25).abs() < 1e-12);
        assert_eq!(contrib[2], 0.0);

        let y2 = array![[1.0, 0.5], [0.5, 1.0]];
        let c2 = hv_contributions(&y2.view(), &reference.view()).unwrap();
        assert!((c2[0] - 0.25).abs() < 1e-12);
        assert!((c2[1] - 0.25).abs() < 1e-12);

        let flat = hypervolume_slice(&[1.0, 0.5, 0.5, 1.0], 2, 2, &[0.0, 0.0]).unwrap();
        assert!((flat - 0.75).abs() < 1e-12);

        let yvar = array![[0.04, 0.0], [0.0, 0.04]];
        let pessimistic = noisy_hv(&y2.view(), Some(&yvar.view()), &reference.view(), 1.0).unwrap();
        assert!(pessimistic < 0.75 && pessimistic > 0.0);
    }
}
