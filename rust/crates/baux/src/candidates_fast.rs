//! Fast RAASP: binomial perturbation count and uniform trust-region draws.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;

use crate::error::ENNError;

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
