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

    fn nearest(&self) -> ([usize; MAX_HISTORY], usize) {
        let mut nearest = [0; MAX_HISTORY];
        let mut count = 0;
        for i in 0..self.distances.len() {
            if Some(i) == self.excluded {
                continue;
            }
            let mut position = count;
            while position > 0 && self.distances[i] < self.distances[nearest[position - 1]] {
                nearest[position] = nearest[position - 1];
                position -= 1;
            }
            nearest[position] = i;
            count += 1;
        }
        (nearest, count.min(self.neighbors))
    }

    fn noise(&self, i: usize) -> f64 {
        let scale_squared = (f64::from(self.scale) * f64::from(self.scale)).max(1.0e-12);
        f64::from(self.aleatoric) + f64::from(self.variances[i]) / scale_squared
    }

    fn precision(&self, i: usize) -> f64 {
        let scale_squared = (f64::from(self.scale) * f64::from(self.scale)).max(1.0e-12);
        1.0 / (1.0e-9
            + f64::from(self.epistemic) * self.distances[i]
            + f64::from(self.aleatoric)
            + f64::from(self.variances[i]) / scale_squared)
            .max(1.0e-12)
    }

    fn law_errors(&self, law: UncertaintyLaw, slice: &[usize], denom: f64) -> (f64, f64) {
        let scale = f64::from(self.scale);
        let epistemic = f64::from(self.epistemic);
        match law {
            UncertaintyLaw::InversePrecision => {
                let mut aleatoric = 0.0_f64;
                for &i in slice {
                    aleatoric += self.precision(i) / denom * self.noise(i);
                }
                let variance = 1.0_f64 / denom;
                (
                    (variance + aleatoric).sqrt() * scale,
                    variance.sqrt() * scale,
                )
            }
            UncertaintyLaw::BiasVariance => {
                let mut weighted_noise = 0.0_f64;
                let mut dist_var = 0.0_f64;
                for &i in slice {
                    let w = self.precision(i);
                    weighted_noise += w * w * self.noise(i);
                    dist_var += (w / denom) * self.distances[i];
                }
                let geom_var = epistemic * dist_var;
                let noise_var = weighted_noise / (denom * denom).max(1.0e-24);
                (
                    (noise_var + geom_var).max(1.0e-12).sqrt() * scale,
                    geom_var.max(0.0).sqrt() * scale,
                )
            }
            UncertaintyLaw::Geometric => {
                let mut dist_var = 0.0_f64;
                for &i in slice {
                    dist_var += (self.precision(i) / denom) * self.distances[i];
                }
                let geom_var = epistemic * dist_var;
                let se = geom_var.max(1.0e-12).sqrt() * scale;
                (se, se)
            }
        }
    }

    pub fn predict_law(&self, law: UncertaintyLaw) -> Result<Prediction, &'static str> {
        self.validate()?;
        let (nearest, mut count) = self.nearest();
        if count == 0 {
            return Ok(Prediction {
                mean: self.outcomes[self.incumbent],
                standard_error: (f64::from(self.aleatoric).max(1.0e-12).sqrt()
                    * f64::from(self.scale)) as f32,
                epistemic_error: 0.0,
                score: self.outcomes[self.incumbent],
                neighbors: 0,
            });
        }
        if self.adaptive {
            let mut prefix = Prefix::default();
            for (rank, &i) in nearest[..count].iter().enumerate() {
                prefix.push(
                    rank as u32 + 1,
                    self.distances[i],
                    f64::from(self.epistemic),
                    self.noise(i),
                );
            }
            count = prefix.count as usize;
        }
        let slice = &nearest[..count];
        let (mut sum, mut value, mut reference) = (0.0_f64, 0.0_f64, f64::MIN_POSITIVE);
        for &i in slice {
            let weight = self.precision(i);
            sum += weight;
            value += weight * f64::from(self.outcomes[i]);
            reference = reference.max(weight);
        }
        let denom = sum.max(1.0e-12);
        let mean = value / denom;
        let (error, epistemic) = self.law_errors(law, slice, denom);
        let score = if let Some(draws) = self.draws {
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
