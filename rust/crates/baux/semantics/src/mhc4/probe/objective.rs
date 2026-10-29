//! Shared objective-reduction fixture, including gains below patch-mean
//! rounding resolution. It is not a forward or optimizer benchmark.
use super::super::{ObjectiveStats, PairShape};

pub struct Fixture {
    pub losses: Vec<f32>,
    pub baseline: Vec<f32>,
    pub generated: Vec<u32>,
    pub targets: Vec<u32>,
    pub shape: PairShape,
    pub expected: ObjectiveStats,
    pub patches: Vec<f32>,
}

impl Fixture {
    pub fn new(tokens: usize, paired: bool) -> Self {
        assert!((1..=1025).contains(&tokens));
        let baseline = (0..tokens)
            .map(|i| 12.0 + ((i * 17) % 37) as f32 * 0.125)
            .collect::<Vec<_>>();
        let losses = baseline
            .iter()
            .enumerate()
            .map(|(i, &v)| v - ((i % 3 + 1) as f32) * (1.0 / 1_048_576.0))
            .collect::<Vec<_>>();
        let targets = (0..tokens)
            .map(|i| (i * 13 % 29) as u32)
            .collect::<Vec<_>>();
        let generated = targets
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 7 == 0 { (v + 1) % 29 } else { v })
            .collect::<Vec<_>>();
        let shape = PairShape {
            tokens: tokens as u32,
            paired: u32::from(paired),
            batch_tokens: 64,
        };
        let mut expected = ObjectiveStats {
            maximum: f32::NEG_INFINITY,
            first_mismatch: u32::MAX,
            batches: tokens.div_ceil(64) as u32,
            ..Default::default()
        };
        for i in 0..tokens {
            expected.sum += losses[i];
            expected.maximum = expected.maximum.max(losses[i]);
            if paired {
                expected.difference_sum += baseline[i] - losses[i];
            }
            if generated[i] == targets[i] {
                expected.matches += 1;
            } else {
                expected.mismatches += 1;
                expected.first_mismatch = expected.first_mismatch.min(i as u32);
                expected.mismatch_sum += losses[i];
            }
        }
        let mut patches = Vec::new();
        for begin in (0..tokens).step_by(64) {
            let end = (begin + 64).min(tokens);
            let (mut sum, mut difference) = (0.0_f32, 0.0_f32);
            for i in begin..end {
                sum += losses[i];
                difference += baseline[i] - losses[i];
            }
            let inverse = 1.0 / (end - begin) as f32;
            let centered = sum * inverse - 12.0;
            expected.batch_centered_sum += centered;
            expected.squared_batch_centered_sum += centered * centered;
            if paired {
                let difference = difference * inverse;
                patches.push(difference);
                expected.difference_batch_sum += difference;
                expected.squared_difference_batch_sum += difference * difference;
            }
        }
        for octile in 0..8 {
            let (mut sum, mut count) = (0.0_f32, 0);
            for base in (0..tokens).step_by(64) {
                for i in base + octile * 8..(base + (octile + 1) * 8).min(tokens) {
                    sum += losses[i];
                    count += 1;
                }
            }
            expected.patch_nll_octiles[octile] = if count == 0 {
                f32::NAN
            } else {
                sum / count as f32
            };
        }
        Self {
            losses,
            baseline,
            generated,
            targets,
            shape,
            expected,
            patches,
        }
    }

    pub fn verify(&self, actual: &ObjectiveStats, patches: &[f32]) -> Result<(), String> {
        let e = &self.expected;
        if (
            actual.batches,
            actual.matches,
            actual.mismatches,
            actual.first_mismatch,
        ) != (e.batches, e.matches, e.mismatches, e.first_mismatch)
        {
            return Err("likelihood reduction integer metadata differs".into());
        }
        // Paired differences are exact multiples of one FP32 ULP in this
        // fixture. Their sum and serial per-patch means must match exactly.
        if actual.difference_sum.to_bits() != e.difference_sum.to_bits() || patches != self.patches
        {
            return Err(format!(
                "paired token reduction differs: gain {} != {}, patches {patches:?} != {:?}",
                actual.difference_sum, e.difference_sum, self.patches
            ));
        }
        let fields = |v: &ObjectiveStats| {
            [
                v.sum,
                v.batch_centered_sum,
                v.squared_batch_centered_sum,
                v.difference_batch_sum,
                v.squared_difference_batch_sum,
                v.maximum,
                v.mismatch_sum,
            ]
        };
        for (a, b) in fields(actual).into_iter().zip(fields(e)).chain(
            actual
                .patch_nll_octiles
                .into_iter()
                .zip(e.patch_nll_octiles),
        ) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            if !a.is_finite() || (a - b).abs() > 64.0 * f32::EPSILON * b.abs().max(1.0) {
                return Err(format!("likelihood moment {a} != {b}"));
            }
        }
        actual.estimate(self.shape)?;
        Ok(())
    }
}
