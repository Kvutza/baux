//! Per-token likelihood and paired patch-moment reporting semantics.
//! The variance is the existing empirical patch estimate, not a concentration
//! bound or a statement about independent tokens/patches.
use baux::semantics::abi::{metal_field, metal_kind, record};

record! {
    PairShape {
        tokens: u32,
        paired: u32,
        batch_tokens: u32,
    }
}

record! {
    ObjectiveStats {
        sum: f32,
        batch_centered_sum: f32,
        squared_batch_centered_sum: f32,
        difference_sum: f32,
        difference_batch_sum: f32,
        squared_difference_batch_sum: f32,
        maximum: f32,
        mismatch_sum: f32,
        batches: u32,
        matches: u32,
        mismatches: u32,
        first_mismatch: u32,
        patch_nll_octiles: [f32; 8],
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossEstimate {
    pub mean: f32,
    pub variance: f32,
    pub bits: f64,
    pub improvement: Option<f32>,
    pub paired_variance: Option<f32>,
}

fn mean_variance(sum: f32, squared: f32, count: u32) -> f32 {
    if count <= 1 {
        return 0.0;
    }
    let n = count as f32;
    (squared - sum * sum / n).max(0.0) / (n * (n - 1.0))
}

impl ObjectiveStats {
    pub fn estimate(&self, shape: PairShape) -> Result<LossEstimate, &'static str> {
        if shape.tokens == 0
            || shape.batch_tokens == 0
            || shape.paired > 1
            || self.batches != shape.tokens.div_ceil(shape.batch_tokens)
            || self.matches.checked_add(self.mismatches) != Some(shape.tokens)
        {
            return Err("likelihood reduction has inconsistent shape/counts");
        }
        if [
            self.sum,
            self.batch_centered_sum,
            self.squared_batch_centered_sum,
            self.difference_sum,
            self.difference_batch_sum,
            self.squared_difference_batch_sum,
            self.maximum,
            self.mismatch_sum,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err("likelihood reduction has nonfinite moments");
        }
        let estimate = LossEstimate {
            mean: self.sum / shape.tokens as f32,
            bits: f64::from(self.sum) / std::f64::consts::LN_2,
            variance: mean_variance(
                self.batch_centered_sum,
                self.squared_batch_centered_sum,
                self.batches,
            ),
            improvement: (shape.paired != 0).then_some(self.difference_sum / shape.tokens as f32),
            paired_variance: (shape.paired != 0).then(|| {
                mean_variance(
                    self.difference_batch_sum,
                    self.squared_difference_batch_sum,
                    self.batches,
                )
            }),
        };
        if !estimate.mean.is_finite()
            || !estimate.variance.is_finite()
            || estimate.improvement.is_some_and(|v| !v.is_finite())
            || estimate.paired_variance.is_some_and(|v| !v.is_finite())
        {
            return Err("likelihood reduction has nonfinite moments");
        }
        Ok(estimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments() {
        let raw = ObjectiveStats {
            sum: 65.0 * 12.5,
            batches: 2,
            matches: 65,
            batch_centered_sum: 1.0,
            squared_batch_centered_sum: 1.0,
            difference_sum: 65.0 * 0.25,
            difference_batch_sum: 0.5,
            squared_difference_batch_sum: 0.25,
            ..Default::default()
        };
        let shape = PairShape {
            tokens: 65,
            paired: 1,
            batch_tokens: 64,
        };
        let estimate = raw.estimate(shape).unwrap();
        assert_eq!(estimate.mean, 12.5);
        assert_eq!(estimate.variance, 0.25);
        assert_eq!(estimate.improvement, Some(0.25));
        assert_eq!(estimate.paired_variance, Some(0.0625));
        assert!(
            raw.estimate(PairShape {
                tokens: 64,
                ..shape
            })
            .is_err()
        );
        assert!(
            ObjectiveStats {
                squared_batch_centered_sum: f32::NAN,
                ..raw
            }
            .estimate(shape)
            .is_err()
        );
        assert_eq!(size_of::<ObjectiveStats>(), 80);
        assert_eq!(core::mem::offset_of!(ObjectiveStats, patch_nll_octiles), 48);
    }
}
