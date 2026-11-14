//! Fast RAASP: binomial perturbation count and uniform trust-region draws.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use rand::Rng;
use rand::distributions::Bernoulli;

use crate::error::ENNError;
use crate::proposal_distribution::CoordSystem;

fn binomial<R: Rng + ?Sized>(rng: &mut R, n: usize, probability: f64) -> usize {
    (0..n).filter(|_| rng.r#gen::<f64>() < probability).count()
}

fn choose<R: Rng + ?Sized>(rng: &mut R, n: usize, count: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for index in 0..count {
        let selected = index + rng.gen_range(0..(n - index));
        indices.swap(index, selected);
    }
    indices.truncate(count);
    indices
}

pub fn generate<R: Rng + ?Sized>(
    center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    candidates: usize,
    rng: &mut R,
    perturbations: usize,
) -> Result<Array2<f64>, ENNError> {
    let dimensions = center.len();
    if dimensions == 0 || candidates == 0 {
        return Ok(Array2::zeros((candidates, dimensions)));
    }
    let probability = (perturbations as f64 / dimensions as f64).min(1.0);
    let mut result = Array2::zeros((candidates, dimensions));
    for row in 0..candidates {
        for column in 0..dimensions {
            result[[row, column]] = center[column];
        }
        let count = binomial(rng, dimensions, probability).max(1);
        for column in choose(rng, dimensions, count) {
            let unit: f64 = rng.r#gen();
            result[[row, column]] = lower[column] + (upper[column] - lower[column]) * unit;
        }
    }
    Ok(result)
}

pub fn build_units(num_dim: usize, tied_dims: &[Vec<usize>]) -> Result<Vec<Vec<usize>>, ENNError> {
    crate::metric_weights::validate_tied(tied_dims, num_dim)?;
    let mut seen = vec![false; num_dim];
    let mut units = Vec::new();
    for group in tied_dims {
        if !group.is_empty() {
            for &d in group {
                seen[d] = true;
            }
            units.push(group.clone());
        }
    }
    for (d, &covered) in seen.iter().enumerate() {
        if !covered {
            units.push(vec![d]);
        }
    }
    Ok(units)
}

pub fn select_units<R: Rng + ?Sized>(
    num_units: usize,
    raasp: bool,
    masked_sobol: bool,
    num_pert: usize,
    rng: &mut R,
) -> Result<Vec<usize>, ENNError> {
    if num_units == 0 {
        return Ok(Vec::new());
    }
    if raasp {
        let k = num_pert.max(1).min(num_units);
        return Ok(choose(rng, num_units, k));
    }
    if masked_sobol {
        let prob = (num_pert as f64 / num_units as f64).min(1.0);
        let bern = Bernoulli::new(prob)
            .map_err(|e| ENNError::InvalidParameter(format!("binomial mask: {e}")))?;
        let k = (0..num_units).filter(|_| rng.sample(bern)).count().max(1);
        return Ok(choose(rng, num_units, k));
    }
    Ok((0..num_units).collect())
}

fn draw_normal<R: Rng + ?Sized>(sobol_val: Option<f64>, rng: &mut R) -> f64 {
    if let Some(u) = sobol_val {
        crate::ndtri::ndtri(u.clamp(1e-12, 1.0 - 1e-12))
    } else {
        let u1 = rng.r#gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
        let u2 = rng.r#gen::<f64>();
        baux::semantics::random::box_muller(u1, u2)
    }
}

fn clamp_box(val: f64, lo: f64, hi: f64) -> f64 {
    if hi > lo { val.clamp(lo, hi) } else { lo }
}

fn origin_step(
    row: &mut ArrayViewMut1<f64>,
    x_center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    dims: &[usize],
    z: &[f64],
) -> bool {
    let center_shift: Vec<f64> = dims.iter().map(|&d| x_center[d] - 0.5).collect();
    let r0 = center_shift.iter().map(|v| v * v).sum::<f64>().sqrt();
    if r0 <= 1e-12 {
        return false;
    }
    let raw_step: Vec<f64> = dims
        .iter()
        .zip(z.iter())
        .map(|(&d, &u)| 0.5 * (upper[d] - lower[d]) * u)
        .collect();
    let dot: f64 = raw_step
        .iter()
        .zip(center_shift.iter())
        .map(|(a, b)| a * b)
        .sum();
    let proj_scale = dot / (r0 * r0);
    let mut tangent: Vec<f64> = raw_step
        .iter()
        .zip(center_shift.iter())
        .map(|(s, c)| s - proj_scale * c)
        .collect();
    if tangent.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1e-12 {
        tangent = raw_step;
    }
    let displaced: Vec<f64> = center_shift
        .iter()
        .zip(tangent.iter())
        .map(|(c, t)| c + t)
        .collect();
    let d_norm = displaced.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if d_norm > 1e-12 { r0 / d_norm } else { 1.0 };
    for (&d, &disp) in dims.iter().zip(displaced.iter()) {
        row[d] = clamp_box(0.5 + disp * scale, lower[d], upper[d]);
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub fn perturb_group<R: Rng + ?Sized>(
    row: &mut ArrayViewMut1<f64>,
    x_center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    dims: &[usize],
    coord_system: CoordSystem,
    sobol_row: Option<&[f64]>,
    rng: &mut R,
) {
    if dims.is_empty() {
        return;
    }
    if coord_system == CoordSystem::Cartesian {
        for &d in dims {
            let u = sobol_row.map_or_else(|| rng.r#gen::<f64>(), |s| s[d]);
            row[d] = clamp_box(lower[d] + u * (upper[d] - lower[d]), lower[d], upper[d]);
        }
        return;
    }
    let mut z: Vec<f64> = dims
        .iter()
        .map(|&d| draw_normal(sobol_row.map(|s| s[d]), rng))
        .collect();
    let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for v in &mut z {
            *v /= norm;
        }
    } else {
        z[0] = 1.0;
    }
    if coord_system == CoordSystem::OriginSphere
        && origin_step(row, x_center, lower, upper, dims, &z)
    {
        return;
    }
    let radial = if coord_system == CoordSystem::Polar {
        rng.r#gen::<f64>()
    } else {
        1.0
    };
    for (&d, &u) in dims.iter().zip(z.iter()) {
        let half = 0.5 * (upper[d] - lower[d]);
        row[d] = clamp_box(x_center[d] + radial * half * u, lower[d], upper[d]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn fixed_trace() {
        let mut rng = StdRng::seed_from_u64(11);
        let values = generate(
            &array![0.5, 0.5, 0.5].view(),
            &array![0.0, 0.0, 0.0],
            &array![1.0, 1.0, 1.0],
            2,
            &mut rng,
            1,
        )
        .unwrap();
        assert_eq!(values.shape(), &[2, 3]);
        assert_eq!(values.iter().filter(|value| **value != 0.5).count(), 2);
    }
}
