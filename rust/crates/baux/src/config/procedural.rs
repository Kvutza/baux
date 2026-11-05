//! Validate logical pool requests before a backend can silently reinterpret
//! them.
use super::RunConfig;
use crate::procedural_pool::ProceduralPool;

impl RunConfig {
    pub fn resident_pool(&self) -> Result<ProceduralPool, String> {
        self.proposal_pool.unwrap_or_default().resolve_resident()
    }
}

#[cfg(test)]
mod tests {
    use crate::config::parse_tune;

    fn document(arms: u32, candidates: u32) -> String {
        format!(
            "version=4\n\
             experiment='end-to-end'\n\
             [optimizer]\n\
             algorithm='turbo-enn'\n\
             arms={arms}\n\
             candidates={candidates}\n\
             acquisition='upper-confidence-bound'\n"
        )
    }

    #[test]
    fn default_pool() {
        let explicit = parse_tune(&document(1, 4)).unwrap();
        assert_eq!(explicit.resident_pool().unwrap().count(), 4);
        assert_eq!(explicit.resident_enn(17).unwrap().ask.acquisition.seed, 17);
    }

    #[test]
    fn invalid_pool() {
        assert!(parse_tune(&document(0, 4)).is_err());
        assert!(parse_tune(&document(1, 0)).is_err());
        assert!(parse_tune(&document(3, 4)).is_err());
    }

    #[test]
    fn fixed_layouts() {
        for (arms, slots) in [(1, 1), (1, 4), (2, 2), (4, 1)] {
            let config = parse_tune(&document(arms, arms * slots)).unwrap();
            let pool = config.resident_enn(0).unwrap().pool;
            assert_eq!(pool.arms(), arms);
            assert_eq!(pool.slots(), slots);
            assert_eq!(pool.count(), arms * slots);
        }
    }
}
