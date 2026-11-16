//! BPANN tuning values.
//!
//! Pure value semantics for tunable parameters. Defaults match historical
//! hardcoded defaults.

/// Minimum allowed `index_max` (matches compaction clamp floor).
pub const INDEX_MIN: usize = 3;

/// Default rows of pending observations before an index flush is scheduled.
pub const PENDING_SOFT: usize = 250;

/// Default hard cap on pending rows before soft sync runs on the calling
/// thread.
///
/// Equal to `4 × PENDING_SOFT`. Must stay `>=` the soft threshold.
pub const PENDING_HARD: usize = 3000;

/// Default max indexed rows for exhaustive leaf search (and no skip edges at
/// build).
pub const EXHAUSTIVE_LIMIT: usize = 2500;

/// Default max indexed rows for skip-refinement search (and skip-edge build).
pub const SKIP_LIMIT: usize = 150_000;

/// Default max batch size for row-id-only leaf builds (mmap score path).
///
/// Matches the historical hardcoded cutoff in `build_batch` on main. Batches
/// larger than this use full in-memory leaf vectors.
pub const STRUCTURED_LIMIT: usize = 1024;

/// Snapshot of tunable BPANN parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpannTuning {
    pub index_fragment: usize,
    pub index_max: usize,
    pub search_fragment: usize,
    pub fragment_rows: usize,
    pub search_limit: usize,
    pub build_seed: Option<u64>,
    pub soft_threshold: usize,
    /// Hard pending cap: soft-sync on the caller when `pending >=` this value.
    pub hard_threshold: usize,
    pub structured_limit: usize,
    pub search_width: usize,
    /// Max indexed rows using exhaustive leaf search; build stores no skip
    /// edges at or below.
    pub exhaustive_limit: usize,
    /// Max indexed rows using skip-refinement search; build stores skip edges
    /// in `(exhaustive_limit, skip_limit]`.
    pub skip_limit: usize,
}

impl Default for BpannTuning {
    fn default() -> Self {
        Self {
            index_fragment: 10_000,
            index_max: 32,
            search_fragment: 80_000,
            fragment_rows: 15_000,
            search_limit: 1,
            build_seed: None,
            soft_threshold: PENDING_SOFT,
            hard_threshold: PENDING_HARD,
            // Batches ≤ this limit use row-id-only leaves (mmap score path).
            structured_limit: STRUCTURED_LIMIT,
            search_width: 1,
            exhaustive_limit: EXHAUSTIVE_LIMIT,
            skip_limit: SKIP_LIMIT,
        }
    }
}

impl BpannTuning {
    /// Construct a tuning configuration with custom soft and hard pending
    /// thresholds.
    pub fn with_thresholds(soft_threshold: usize, hard_threshold: usize) -> Self {
        Self {
            soft_threshold,
            hard_threshold,
            ..Self::default()
        }
    }

    /// Construct a tuning configuration with custom exhaustive and skip row
    /// limits.
    pub fn with_limits(exhaustive_limit: usize, skip_limit: usize) -> Self {
        Self {
            exhaustive_limit,
            skip_limit,
            ..Self::default()
        }
    }

    /// Validate all tunable fields. Returns an error describing the first
    /// violation.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            (
                self.index_fragment == 0,
                "index_fragment must be >= 1".to_string(),
            ),
            (
                self.index_max < INDEX_MIN,
                format!("index_max must be >= {INDEX_MIN}"),
            ),
            (
                self.search_fragment == 0,
                "search_fragment must be >= 1".to_string(),
            ),
            (
                self.fragment_rows == 0,
                "fragment_rows must be >= 1".to_string(),
            ),
            (
                self.search_limit == 0,
                "search_limit must be >= 1".to_string(),
            ),
            (
                self.soft_threshold == 0,
                "soft_threshold must be >= 1".to_string(),
            ),
            (
                self.hard_threshold == 0,
                "hard_threshold must be >= 1".to_string(),
            ),
            (
                self.hard_threshold < self.soft_threshold,
                "hard_threshold must be >= soft_threshold".to_string(),
            ),
            (
                self.structured_limit == 0,
                "structured_limit must be >= 1".to_string(),
            ),
            (
                self.search_width == 0,
                "search_width must be >= 1".to_string(),
            ),
            (
                self.exhaustive_limit == 0,
                "exhaustive_limit must be >= 1".to_string(),
            ),
            (
                self.skip_limit < self.exhaustive_limit,
                "skip_limit must be >= exhaustive_limit".to_string(),
            ),
        ];
        for (invalid, message) in checks {
            if invalid {
                return Err(message);
            }
        }
        Ok(())
    }

    /// Whether build should store skip edges for this indexed-row count.
    ///
    /// On-disk skip edges reflect build-time limits; search uses call-time
    /// limits and may take the skip-refinement path with empty edges until
    /// rebuild.
    pub fn rows_edges(&self, row_count: usize) -> bool {
        row_count > self.exhaustive_limit && row_count <= self.skip_limit
    }

    /// Whether search should scan all leaves exhaustively for this row count.
    pub fn exhaustive_search(&self, rows: usize) -> bool {
        rows <= self.exhaustive_limit
    }

    /// Whether search should use skip-refinement (non-exhaustive, within skip
    /// band).
    pub fn refine_search(&self, rows: usize) -> bool {
        !self.exhaustive_search(rows) && rows <= self.skip_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning() {
        assert!(BpannTuning::default().validate().is_ok());
    }

    #[test]
    fn pending_250() {
        assert_eq!(PENDING_SOFT, 250);
        assert_eq!(BpannTuning::default().soft_threshold, PENDING_SOFT);
        let t = BpannTuning::with_thresholds(PENDING_SOFT, PENDING_HARD);
        assert_eq!(t.soft_threshold, PENDING_SOFT);
    }

    #[test]
    fn pending_3000() {
        assert_eq!(PENDING_HARD, 3000);
        // hard >= soft is enforced by BpannTuning::validate (see tuning).
        assert_eq!(BpannTuning::default().hard_threshold, PENDING_HARD);
        let t = BpannTuning::with_thresholds(PENDING_SOFT, PENDING_HARD);
        assert_eq!(t.hard_threshold, PENDING_HARD);
    }

    #[test]
    fn metamorphic_fields() {
        // Changing unrelated fields must not change the pending_flush default.
        let base = BpannTuning::default();
        let variants = [
            BpannTuning {
                index_fragment: 5_000,
                ..base
            },
            BpannTuning {
                search_width: 8,
                ..base
            },
            BpannTuning {
                structured_limit: 4_096,
                ..base
            },
        ];
        for v in variants {
            assert_eq!(v.soft_threshold, PENDING_SOFT);
            assert_eq!(v.hard_threshold, PENDING_HARD);
            assert!(v.validate().is_ok());
        }
    }

    #[test]
    fn fuzz_seeds2() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = 0x5045_4e44_u64; // "PEND"
        println!("fuzz_pending_threshold_validation seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..64 {
            let soft = rng.gen_range(0usize..10_000);
            let hard = rng.gen_range(0usize..10_000);
            let t = BpannTuning::with_thresholds(soft, hard);
            let expect_ok = soft >= 1 && hard >= 1 && hard >= soft;
            assert_eq!(t.validate().is_ok(), expect_ok, "soft={soft} hard={hard}");
        }
    }

    #[test]
    fn rejects_threshold() {
        let t = BpannTuning::with_thresholds(500, 499);
        assert!(t.validate().unwrap_err().contains("hard_threshold"));
    }

    #[test]
    fn rejects_threshold2() {
        let t = BpannTuning::with_thresholds(1, 0);
        assert!(t.validate().unwrap_err().contains("hard_threshold"));
    }

    #[test]
    fn rejects_beam() {
        let t = BpannTuning {
            index_fragment: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("index_fragment"));
        let t = BpannTuning {
            search_fragment: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("search_fragment"));
        let t = BpannTuning {
            search_width: 0,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("search_width"));
    }

    #[test]
    fn rejects_floor() {
        let t = BpannTuning {
            index_max: INDEX_MIN - 1,
            ..Default::default()
        };
        assert!(t.validate().unwrap_err().contains("index_max"));
    }

    #[test]
    fn tuning_constructors() {
        let t = BpannTuning::with_thresholds(100, 500);
        assert_eq!(t.soft_threshold, 100);
        assert_eq!(t.hard_threshold, 500);
        assert!(t.validate().is_ok());

        let t_limits = BpannTuning::with_limits(1000, 50_000);
        assert_eq!(t_limits.exhaustive_limit, 1000);
        assert_eq!(t_limits.skip_limit, 50_000);
        assert!(t_limits.validate().is_ok());
    }

    #[test]
    fn search_cliffs() {
        let t = BpannTuning::default();
        assert_eq!(t.exhaustive_limit, EXHAUSTIVE_LIMIT);
        assert_eq!(t.skip_limit, SKIP_LIMIT);
        assert_eq!(EXHAUSTIVE_LIMIT, 2500);
        assert_eq!(SKIP_LIMIT, 150_000);
        assert_eq!(t.structured_limit, STRUCTURED_LIMIT);
        assert_eq!(STRUCTURED_LIMIT, 1024);
        // Latency default: search at most one fragment when many exist
        // (proposal scout).
        assert_eq!(t.search_limit, 1);
    }

    #[test]
    fn needs_boundaries() {
        let t = BpannTuning::default();
        assert!(!t.rows_edges(0));
        assert!(!t.rows_edges(EXHAUSTIVE_LIMIT));
        assert!(t.rows_edges(EXHAUSTIVE_LIMIT + 1));
        assert!(t.rows_edges(SKIP_LIMIT));
        assert!(!t.rows_edges(SKIP_LIMIT + 1));
    }

    #[test]
    fn rejects_exhaustive() {
        let t = BpannTuning::with_limits(0, SKIP_LIMIT);
        assert!(t.validate().unwrap_err().contains("exhaustive_limit"));
        let t = BpannTuning::with_limits(100, 99);
        assert!(t.validate().unwrap_err().contains("skip_limit"));
    }

    #[test]
    fn equal_band() {
        let t = BpannTuning::with_limits(500, 500);
        assert!(t.validate().is_ok());
        assert!(!t.rows_edges(500));
        assert!(!t.rows_edges(501));
        assert!(t.exhaustive_search(500));
        assert!(!t.refine_search(501));
    }

    #[test]
    fn metamorphic_edges() {
        // For every valid limit pair and row count, skip-edge build iff
        // skip-refine search.
        let base = BpannTuning::default();
        let pairs = [
            (1usize, 1usize),
            (1, 10),
            (2500, 150_000),
            (100, 100),
            (10_000, usize::MAX),
        ];
        for (ex, skip) in pairs {
            let t = BpannTuning::with_limits(ex, skip);
            assert!(t.validate().is_ok(), "ex={ex} skip={skip}");
            for rows in [
                0,
                1,
                ex.saturating_sub(1),
                ex,
                ex.saturating_add(1),
                skip,
                skip.saturating_add(1),
            ] {
                assert_eq!(
                    t.rows_edges(rows),
                    t.refine_search(rows),
                    "rows={rows} ex={ex} skip={skip}"
                );
                assert_eq!(
                    t.exhaustive_search(rows),
                    rows <= ex,
                    "exhaustive rows={rows}"
                );
            }
        }
    }

    #[test]
    fn fuzz_seeds3() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = 0x524f_5753_u64; // "ROWS"
        println!("fuzz_search_row_limit_validation seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..128 {
            let ex = rng.gen_range(0usize..5_000);
            let skip = rng.gen_range(0usize..10_000);
            let t = BpannTuning::with_limits(ex, skip);
            let ok = t.validate().is_ok();
            if ex == 0 || skip < ex {
                assert!(!ok, "ex={ex} skip={skip} should be invalid");
            } else {
                assert!(ok, "ex={ex} skip={skip} should be valid");
            }
        }
    }

    #[test]
    fn tuning_rows() {
        let rows = 3_000usize;
        let default_tuning = BpannTuning::default();
        // Defaults: 3000 is in the skip-refinement band.
        assert!(default_tuning.rows_edges(rows));
        assert!(default_tuning.refine_search(rows));

        let t1 = BpannTuning::with_limits(10_000, 150_000);
        assert!(!t1.rows_edges(rows));
        assert!(t1.exhaustive_search(rows));

        let t2 = BpannTuning::with_limits(100, 200);
        assert!(!t2.rows_edges(rows));
        assert!(!t2.exhaustive_search(rows));
        assert!(!t2.refine_search(rows));
    }
}
