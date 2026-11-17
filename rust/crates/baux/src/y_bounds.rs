//! Optional per-metric output bounds for baux.

use std::path::Path;

use ndarray::{Array2, Array3, ArrayView2};

use crate::error::ENNError;

pub fn unbounded_bounds(metrics: usize) -> Array2<f64> {
    Array2::from_shape_fn((metrics, 2), |(_, j)| {
        if j == 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    })
}

pub fn identity_bounds(bounds: &Array2<f64>) -> bool {
    bounds
        .rows()
        .into_iter()
        .all(|r| r[0] == f64::NEG_INFINITY && r[1] == f64::INFINITY)
}

pub fn validate_bounds(bounds: &Array2<f64>, metrics: usize) -> Result<(), ENNError> {
    if bounds.dim() != (metrics, 2) {
        return Err(ENNError::InvalidShape {
            expected: vec![metrics, 2],
            got: bounds.shape().to_vec(),
        });
    }
    for i in 0..metrics {
        let (a, b) = (bounds[[i, 0]], bounds[[i, 1]]);
        if a.is_nan() || b.is_nan() || (a.is_finite() && b.is_finite() && a >= b) {
            return Err(ENNError::InvalidParameter(format!(
                "invalid y_bounds row {i}: ({a}, {b})"
            )));
        }
        if (!a.is_finite() && a != f64::NEG_INFINITY) || (!b.is_finite() && b != f64::INFINITY) {
            return Err(ENNError::InvalidParameter(format!(
                "open y_bounds must use -inf/+inf at row {i}"
            )));
        }
    }
    Ok(())
}

#[inline(always)]
pub fn warp_scalar(y: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::warp_scalar(y, a, b)
}

#[inline(always)]
pub fn inv_scalar(z: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::inv_scalar(z, a, b)
}

#[inline(always)]
pub fn jac_inv(z: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::jac_inv(z, a, b)
}

#[inline(always)]
pub fn warp_var(y: f64, yvar: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::warp_var(y, yvar, a, b).unwrap_or(f64::NAN)
}

#[inline(always)]
pub fn inv_var(z: f64, zvar: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::natural_var(z, zvar, a, b)
}

#[inline(always)]
pub fn project_mu(mu: f64, a: f64, b: f64) -> f64 {
    baux::semantics::ybounds::project_mu(mu, a, b)
}

/// Per-metric invertible output-space transformation between natural `y` and
/// unconstrained `z`.
pub trait OutputTransform: Send + Sync {
    fn metrics(&self) -> usize;
    fn is_identity(&self) -> bool;
    fn warp_val(&self, metric: usize, y: f64) -> Result<f64, ENNError>;
    fn inv_val(&self, metric: usize, z: f64) -> f64;
    fn jac_val(&self, metric: usize, z: f64) -> f64;
    fn project_val(&self, metric: usize, y: f64) -> f64;
}

impl OutputTransform for Array2<f64> {
    fn metrics(&self) -> usize {
        self.nrows()
    }

    fn is_identity(&self) -> bool {
        identity_bounds(self)
    }

    fn warp_val(&self, metric: usize, y: f64) -> Result<f64, ENNError> {
        let (a, b) = (self[[metric, 0]], self[[metric, 1]]);
        if !y.is_finite() || (a.is_finite() && y <= a) || (b.is_finite() && y >= b) {
            return Err(ENNError::InvalidParameter(format!(
                "y[:,{metric}] is outside its open bounds"
            )));
        }
        Ok(warp_scalar(y, a, b))
    }

    fn inv_val(&self, metric: usize, z: f64) -> f64 {
        inv_scalar(z, self[[metric, 0]], self[[metric, 1]])
    }

    fn jac_val(&self, metric: usize, z: f64) -> f64 {
        jac_inv(z, self[[metric, 0]], self[[metric, 1]])
    }

    fn project_val(&self, metric: usize, y: f64) -> f64 {
        project_mu(y, self[[metric, 0]], self[[metric, 1]])
    }
}

impl<T: OutputTransform + ?Sized> OutputTransform for std::sync::Arc<T> {
    fn metrics(&self) -> usize {
        (**self).metrics()
    }

    fn is_identity(&self) -> bool {
        (**self).is_identity()
    }

    fn warp_val(&self, metric: usize, y: f64) -> Result<f64, ENNError> {
        (**self).warp_val(metric, y)
    }

    fn inv_val(&self, metric: usize, z: f64) -> f64 {
        (**self).inv_val(metric, z)
    }

    fn jac_val(&self, metric: usize, z: f64) -> f64 {
        (**self).jac_val(metric, z)
    }

    fn project_val(&self, metric: usize, y: f64) -> f64 {
        (**self).project_val(metric, y)
    }
}

pub fn warp_y(y: ArrayView2<f64>, bounds: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    validate_bounds(bounds, y.ncols())?;
    let mut out = Array2::zeros(y.raw_dim());
    for ((i, j), value) in y.indexed_iter() {
        let (a, b) = (bounds[[j, 0]], bounds[[j, 1]]);
        if !value.is_finite() || (a.is_finite() && *value <= a) || (b.is_finite() && *value >= b) {
            return Err(ENNError::InvalidParameter(format!(
                "y[{i},{j}] is outside its open bounds"
            )));
        }
        out[[i, j]] = warp_scalar(*value, a, b);
    }
    Ok(out)
}

pub fn inv_y(z: ArrayView2<f64>, bounds: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(z.raw_dim(), |(i, j)| {
        inv_scalar(z[[i, j]], bounds[[j, 0]], bounds[[j, 1]])
    })
}

pub fn warp_yvar(
    y: ArrayView2<f64>,
    yvar: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Result<Array2<f64>, ENNError> {
    if y.raw_dim() != yvar.raw_dim() {
        return Err(ENNError::InvalidShape {
            expected: y.shape().to_vec(),
            got: yvar.shape().to_vec(),
        });
    }
    validate_bounds(bounds, y.ncols())?;
    let mut out = Array2::zeros(y.raw_dim());
    for ((i, j), value) in y.indexed_iter() {
        let v = warp_var(*value, yvar[[i, j]], bounds[[j, 0]], bounds[[j, 1]]);
        if !v.is_finite() {
            return Err(ENNError::InvalidParameter(format!(
                "warped yvar[{i},{j}] is not finite"
            )));
        }
        out[[i, j]] = v;
    }
    Ok(out)
}

pub fn naturalize_yvar(
    z: ArrayView2<f64>,
    zvar: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Array2<f64> {
    Array2::from_shape_fn(z.raw_dim(), |(i, j)| {
        inv_var(z[[i, j]], zvar[[i, j]], bounds[[j, 0]], bounds[[j, 1]])
    })
}

fn bounds_json(bounds: &Array2<f64>) -> String {
    let rows = bounds
        .rows()
        .into_iter()
        .map(|row| {
            let value = |v: f64| {
                if v.is_finite() {
                    v.to_string()
                } else {
                    "null".to_string()
                }
            };
            format!("[{},{}]", value(row[0]), value(row[1]))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rows}]")
}

pub fn persist_metadata(work_dir: &Path, bounds: &Array2<f64>) -> Result<(), ENNError> {
    let path = work_dir.join("metadata.json");
    if !path.exists() {
        return Ok(());
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let field = format!("\"y_bounds\":{}", bounds_json(bounds));
    let updated = if let Some(start) = text.find("\"y_bounds\":") {
        let array_start = text[start..]
            .find('[')
            .map(|v| start + v)
            .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
        let mut depth = 0usize;
        let mut array_end = None;
        for (offset, ch) in text[array_start..].char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        array_end = Some(array_start + offset + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = array_end
            .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
        format!("{}{}{}", &text[..start], field, &text[end..])
    } else {
        let end = text
            .rfind('}')
            .ok_or_else(|| ENNError::InvalidParameter("malformed metadata.json".to_string()))?;
        let comma = if text[..end].trim_end().ends_with('{') {
            ""
        } else {
            ","
        };
        format!("{}{}{}{}", &text[..end], comma, field, &text[end..])
    };
    std::fs::write(path, updated).map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn load_metadata(work_dir: &Path, metrics: usize) -> Result<Option<Array2<f64>>, ENNError> {
    let path = work_dir.join("metadata.json");
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let Some(field_start) = text.find("\"y_bounds\":") else {
        return Ok(None);
    };
    let start = text[field_start..]
        .find('[')
        .map(|v| field_start + v)
        .ok_or_else(|| ENNError::InvalidParameter("malformed y_bounds metadata".to_string()))?;
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut depth = 0usize;
    for ch in text[start..].chars() {
        match ch {
            '[' => depth += 1,
            ']' => {
                if !token.trim().is_empty() {
                    tokens.push(token.trim().to_string());
                    token.clear();
                }
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' => {
                if !token.trim().is_empty() {
                    tokens.push(token.trim().to_string());
                    token.clear();
                }
            }
            c if !c.is_whitespace() => token.push(c),
            _ => {}
        }
    }
    if tokens.len() != metrics * 2 {
        return Err(ENNError::InvalidParameter(
            "persisted y_bounds shape mismatch".to_string(),
        ));
    }
    let mut bounds = Array2::zeros((metrics, 2));
    for i in 0..metrics {
        bounds[[i, 0]] = if tokens[2 * i] == "null" {
            f64::NEG_INFINITY
        } else {
            tokens[2 * i]
                .parse()
                .map_err(|_| ENNError::InvalidParameter("invalid y_bounds metadata".to_string()))?
        };
        bounds[[i, 1]] = if tokens[2 * i + 1] == "null" {
            f64::INFINITY
        } else {
            tokens[2 * i + 1]
                .parse()
                .map_err(|_| ENNError::InvalidParameter("invalid y_bounds metadata".to_string()))?
        };
    }
    validate_bounds(&bounds, metrics)?;
    Ok(Some(bounds))
}

pub fn naturalize(
    mu: &mut Array2<f64>,
    se: &mut Array2<f64>,
    se_epi: &mut Array2<f64>,
    se_ale: &mut Array2<f64>,
    bounds: &Array2<f64>,
) {
    if identity_bounds(bounds) {
        return;
    }
    for ((i, j), value) in mu.indexed_iter_mut() {
        let (m, s, e, a) = baux::semantics::ybounds::natural_point(
            *value,
            se[[i, j]],
            se_epi[[i, j]],
            se_ale[[i, j]],
            bounds[[j, 0]],
            bounds[[j, 1]],
        );
        *value = m;
        se[[i, j]] = s;
        se_epi[[i, j]] = e;
        se_ale[[i, j]] = a;
    }
}

pub fn naturalize_batch(
    mu: &mut Array3<f64>,
    se: &mut Array3<f64>,
    se_epi: &mut Array3<f64>,
    se_ale: &mut Array3<f64>,
    bounds: &Array2<f64>,
) {
    if identity_bounds(bounds) {
        return;
    }
    for ((p, i, j), value) in mu.indexed_iter_mut() {
        let (m, s, e, a) = baux::semantics::ybounds::natural_point(
            *value,
            se[[p, i, j]],
            se_epi[[p, i, j]],
            se_ale[[p, i, j]],
            bounds[[j, 0]],
            bounds[[j, 1]],
        );
        *value = m;
        se[[p, i, j]] = s;
        se_epi[[p, i, j]] = e;
        se_ale[[p, i, j]] = a;
    }
}

/// Pure value-returning output naturalization over `(mu, se, se_epi, se_ale)`.
pub fn naturalized(
    mut mu: Array2<f64>,
    mut se: Array2<f64>,
    mut se_epi: Array2<f64>,
    mut se_ale: Array2<f64>,
    bounds: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    naturalize(&mut mu, &mut se, &mut se_epi, &mut se_ale, bounds);
    (mu, se, se_epi, se_ale)
}

/// Pure value-returning inverse warp for posterior sample tensors.
pub fn mapped_draws(mut draws: Array3<f64>, bounds: &Array2<f64>) -> Array3<f64> {
    inverse_draws(&mut draws, bounds);
    draws
}

pub fn inverse_draws(draws: &mut Array3<f64>, bounds: &Array2<f64>) {
    if identity_bounds(bounds) {
        return;
    }
    let metrics = draws.shape()[2];
    assert_eq!(metrics, bounds.nrows());
    for ((_, _, j), value) in draws.indexed_iter_mut() {
        *value = inv_scalar(*value, bounds[[j, 0]], bounds[[j, 1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bounded_trip2() {
        let bounds = array![[0.0, 1.0], [0.0, f64::INFINITY]];
        let y = array![[0.25, 2.0], [0.75, 5.0]];
        let z = warp_y(y.view(), &bounds).unwrap();
        let restored = inv_y(z.view(), &bounds);
        assert!((restored - y).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn rejects_interval() {
        let bounds = array![[0.0, 1.0]];
        assert!(warp_y(array![[0.0]].view(), &bounds).is_err());
        assert!(warp_y(array![[1.0]].view(), &bounds).is_err());
    }
}
