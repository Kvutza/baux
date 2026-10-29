//! Causal corpus-window indexing shared by the accelerator dispatchers.
use super::{PATCH, WaveShape, shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalPlan {
    pub rows: usize,
    pub start: usize,
    pub targets: usize,
}

impl CausalPlan {
    /// Coarse row zero consumes input patch zero, then predicts label 63.
    /// The first 63 labels have no preceding coarse state and are not scored.
    pub fn new(tokens: usize, labels: usize, mask: &[u8], capacity: usize) -> Result<Self, String> {
        if tokens < PATCH
            || labels != tokens
            || mask.len() != tokens
            || mask.iter().any(|&v| v != 1)
        {
            return Err(
                "causal mHC-4 batch requires matching token/label lengths, at least one patch and \
                 an all-one loss mask"
                    .into(),
            );
        }
        Self::window(tokens, capacity)
    }

    pub fn window(tokens: usize, capacity: usize) -> Result<Self, String> {
        if tokens < PATCH {
            return Err("causal mHC-4 needs at least one complete input patch".into());
        }
        let rows = tokens
            .div_ceil(PATCH)
            .checked_next_multiple_of(128)
            .ok_or("causal row count overflow")?;
        if rows > capacity {
            return Err("causal mHC-4 batch exceeds the coarse context".into());
        }
        let plan = Self {
            rows,
            start: PATCH - 1,
            targets: tokens - (PATCH - 1),
        };
        plan.wave()?;
        Ok(plan)
    }

    pub fn wave(self) -> Result<WaveShape, String> {
        shape(PATCH, self.targets, 0.0, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing() {
        for tokens in [64, 65, 128, 8_192, 1_048_576] {
            let plan = CausalPlan::new(tokens, tokens, &vec![1; tokens], 16_384).unwrap();
            assert_eq!(plan.start, 63);
            assert_eq!(plan.targets, tokens - 63);
            assert_eq!(plan.rows, tokens.div_ceil(64).next_multiple_of(128));
            assert_eq!(plan.wave().unwrap().proposal_start, 0);
            assert_eq!(plan.wave().unwrap().tokens as usize, tokens - 63);
        }
        assert!(CausalPlan::new(63, 63, &[1; 63], 128).is_err());
        assert!(CausalPlan::new(64, 63, &[1; 64], 128).is_err());
        assert!(CausalPlan::new(64, 64, &[0; 64], 128).is_err());
        assert!(CausalPlan::new(64, 64, &[1; 64], 64).is_err());
    }
}
