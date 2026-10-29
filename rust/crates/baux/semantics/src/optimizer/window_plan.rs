//! Verified window schedule and score-receipt conservation laws.
//!
//! Enforces that any executed window evaluation in `baux` covers 100% of
//! the planned window tokens (`prefilled + scored == planned`) with zero
//! skipped middle chunks and positive scored tokens.

/// Contiguous window execution plan for causal pretraining evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    pub planned: u64,
    pub chunk: u64,
    pub prefilled: u64,
    pub scored: u64,
}

impl WindowPlan {
    /// Construct a full-window scoring plan where 100% of `planned` tokens are
    /// causally evaluated and scored in contiguous chunks of `chunk` capacity.
    pub fn full(planned: u64, chunk: u64) -> Result<Self, &'static str> {
        if planned == 0 || chunk == 0 {
            return Err("window plan requires positive token and chunk counts");
        }
        Ok(Self {
            planned,
            chunk,
            prefilled: 0,
            scored: planned,
        })
    }

    /// Number of contiguous chunks required to evaluate the window.
    pub fn chunks(self) -> u64 {
        self.planned.div_ceil(self.chunk)
    }
}

/// Check that an executed window covered 100% of `planned` tokens
/// (`prefilled + scored == planned`) with at least one scored token.
pub fn verify_coverage(planned: u64, prefilled: u64, scored: u64) -> bool {
    if planned == 0 || scored == 0 {
        return false;
    }
    match prefilled.checked_add(scored) {
        Some(total) => total == planned,
        None => false,
    }
}

/// Check that a full-window score evaluated and scored 100% of `planned`
/// (`prefilled == 0 && scored == planned`).
pub fn verify_full(planned: u64, prefilled: u64, scored: u64) -> bool {
    planned > 0 && prefilled == 0 && scored == planned
}

/// Advance the cumulative scored-token clock if and only if the window receipt
/// satisfies full window coverage.
pub fn advance_clock(prior: u64, planned: u64, prefilled: u64, scored: u64) -> Option<u64> {
    if !verify_coverage(planned, prefilled, scored) {
        return None;
    }
    prior.checked_add(scored)
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn prove_coverage() {
        let planned: u64 = kani::any();
        let prefilled: u64 = kani::any();
        let scored: u64 = kani::any();
        let ok = verify_coverage(planned, prefilled, scored);
        if ok {
            assert!(planned > 0);
            assert!(scored > 0);
            assert_eq!(prefilled.checked_add(scored), Some(planned));
        }
    }

    #[kani::proof]
    fn prove_shortcut() {
        assert!(!verify_coverage(1_048_576, 2_048, 1_024));
        assert!(!verify_full(1_048_576, 2_048, 1_024));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_laws() {
        assert!(verify_coverage(1_048_576, 0, 1_048_576));
        assert!(verify_full(1_048_576, 0, 1_048_576));
        assert!(!verify_coverage(1_048_576, 2_048, 1_024));
        assert!(!verify_full(1_048_576, 2_048, 1_024));
        assert!(!verify_coverage(0, 0, 0));
        assert!(!verify_coverage(4096, 4096, 0));
        assert_eq!(advance_clock(100, 4096, 0, 4096), Some(4196));
        assert_eq!(advance_clock(100, 1_048_576, 2_048, 1_024), None);
        let plan = WindowPlan::full(1_048_576, 4096).unwrap();
        assert_eq!(plan.chunks(), 256);
    }
}
