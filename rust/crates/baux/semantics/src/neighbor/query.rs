//! FP32 reference for the resident Metal posterior and acquisition operation.
//! Order of accumulation and first-index tie breaking are part of this rule.
use super::{MAX_HISTORY, Prefix, UncertaintyLaw};

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

fn law_errors(
    law: UncertaintyLaw,
    slice: &[usize],
    distances: &[f64],
    variances: &[f32],
    epistemic: f64,
    aleatoric: f64,
    scale: f64,
    denom: f64,
) -> (f64, f64) {
    let scale_sq = (scale * scale).max(1.0e-12);
    let inv_denom = 1.0_f64 / denom;
    let noise = |i: usize| aleatoric + f64::from(variances[i]) / scale_sq;
    let prec = |i: usize| {
        1.0 / (1.0e-9 + epistemic * distances[i] + aleatoric + f64::from(variances[i]) / scale_sq)
            .max(1.0e-12)
    };
    match law {
        UncertaintyLaw::InversePrecision => {
            let ale: f64 = slice
                .iter()
                .map(|&i| (prec(i) * inv_denom) * noise(i))
                .sum();
            let variance = inv_denom;
            (
                (variance + ale).max(1.0e-12).sqrt() * scale,
                variance.sqrt() * scale,
            )
        }
        UncertaintyLaw::BiasVariance => {
            let (mut weighted_noise, mut dist_var) = (0.0_f64, 0.0_f64);
            for &i in slice {
                let w = prec(i);
                weighted_noise += w * w * noise(i);
                dist_var += (w * inv_denom) * distances[i];
            }
            let geom_var = epistemic * dist_var;
            let noise_var = weighted_noise / (denom * denom).max(1.0e-24);
            (
                (noise_var + geom_var).max(1.0e-12).sqrt() * scale,
                geom_var.max(0.0).sqrt() * scale,
            )
        }
        UncertaintyLaw::Geometric => {
            let dist_var: f64 = slice
                .iter()
                .map(|&i| (prec(i) * inv_denom) * distances[i])
                .sum();
            let se = (epistemic * dist_var).max(1.0e-12).sqrt() * scale;
            (se, se)
        }
    }
}

/// Zero-allocation FP64 precision-weighted neighbor posterior returning
/// `(mean, standard_error, epistemic_error, neighbor_count)`.
#[allow(clippy::too_many_arguments)]
pub fn posterior_f64(
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
) -> (f64, f64, f64, usize) {
    let (nearest, mut count) = nearest_indices(distances, neighbors, excluded);
    if count == 0 {
        let fallback = excluded.unwrap_or(0).min(outcomes.len().saturating_sub(1));
        return (
            outcomes.get(fallback).copied().map_or(0.0, f64::from),
            aleatoric.max(1.0e-12).sqrt() * scale,
            0.0,
            0,
        );
    }
    let scale_squared = (scale * scale).max(1.0e-12);
    let noise = |i: usize| aleatoric + f64::from(variances[i]) / scale_squared;
    let precision = |i: usize| {
        1.0 / (1.0e-9
            + epistemic * distances[i]
            + aleatoric
            + f64::from(variances[i]) / scale_squared)
            .max(1.0e-12)
    };
    if adaptive {
        let mut prefix = Prefix::default();
        for (rank, &i) in nearest[..count].iter().enumerate() {
            prefix.push(rank as u32 + 1, distances[i], epistemic, noise(i));
        }
        count = prefix.count as usize;
    }
    let slice = &nearest[..count];
    let sum: f64 = slice.iter().map(|&i| precision(i)).sum();
    let denom = sum.max(1.0e-12);
    let inv_denom = 1.0_f64 / denom;
    let mean: f64 = slice
        .iter()
        .map(|&i| (precision(i) * inv_denom) * f64::from(outcomes[i]))
        .sum();
    let (error, epistemic_error) = law_errors(
        law, slice, distances, variances, epistemic, aleatoric, scale, denom,
    );
    (mean, error, epistemic_error, count)
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
        let (mean, error, epistemic, count) = posterior_f64(
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
            let (mut noise, mut squared) = (0.0_f64, 0.0_f64);
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
