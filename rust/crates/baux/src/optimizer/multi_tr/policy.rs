//! Independent policy choices for regional search.
//!
//! Eriksson et al., "Scalable Global Optimization via Local Bayesian
//! Optimization," NeurIPS 2019, motivate multiple local models with global
//! sample allocation: https://proceedings.neurips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html.
//! MTRBO's region roles motivate making numerical method and acquisition
//! choices explicit (Das et al., 2026, https://arxiv.org/abs/2605.06618).
//! The concrete variants below are BAUX rules; they are not claimed to be the
//! cited papers' algorithms.
//! APOSMM's start separation is a related software reference:
//! https://github.com/Libensemble/libensemble/blob/2f79ba6943083ff71875272d6f3f9b5222acd273/libensemble/gen_funcs/persistent_aposmm.py.
//! Candidate maximin below is not APOSMM's full start-eligibility rule.

use ndarray::{Array1, ArrayView1};
use rand::RngCore;

use super::{MultiTrustRegionState, SharingPolicy};
use crate::trust_region::TrustRegionError;

/// How candidates for a new region center are ranked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartCenterSelection {
    /// Choose the first unit-cube draw. Further draws permit matched random
    /// consumption when comparing this rule with another ranking.
    First,
    /// Choose the candidate farthest from its nearest active region center.
    Maximin,
}

/// Bounded candidate search for an automatically retired region slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartCenterPolicy {
    pub candidates: usize,
    pub selection: RestartCenterSelection,
}

impl Default for RestartCenterPolicy {
    fn default() -> Self {
        Self {
            candidates: 1,
            selection: RestartCenterSelection::First,
        }
    }
}

impl MultiTrustRegionState {
    /// Sample a replacement center using this state's configured policy.
    /// The first candidate is used when no other slot is active; candidates
    /// are always fully drawn so paired selection rules consume the same RNG.
    pub fn choose_center(
        &self,
        region: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array1<f64>, TrustRegionError> {
        if region >= self.num_regions {
            return Err(TrustRegionError::InvalidParameter(
                "restart region is out of bounds".into(),
            ));
        }
        let policy = self.config.restart_center_policy;
        let mut best = self.draw_center(rng);
        let mut best_distance = self.nearest_distance(region, &best);
        for _ in 1..policy.candidates {
            let candidate = self.draw_center(rng);
            if policy.selection == RestartCenterSelection::Maximin {
                let distance = self.nearest_distance(region, &candidate);
                if distance > best_distance {
                    best = candidate;
                    best_distance = distance;
                }
            }
        }
        Ok(best)
    }

    fn draw_center(&self, rng: &mut dyn RngCore) -> Array1<f64> {
        Array1::from_shape_fn(self.num_dim, |_| rand::Rng::gen_range(rng, 0.0..1.0))
    }

    fn nearest_distance(&self, region: usize, candidate: &Array1<f64>) -> f64 {
        (0..self.num_regions)
            .filter(|&slot| slot != region && self.active_mask[slot])
            .map(|slot| {
                candidate
                    .iter()
                    .zip(self.centers.row(slot))
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Policy used to divide a candidate budget among active region slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPolicy {
    /// Greedily allocate using objective utility and exploration uncertainty.
    UtilityUcb,
    /// Favor the least-evaluated active slots, with stable slot-order ties.
    Balanced,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        Self::UtilityUcb
    }
}

/// Policy for regional observations when a slot starts a new generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMemoryPolicy {
    /// Start with an empty local model, as a new search trajectory.
    Fresh,
    /// Refit the new generation on observations previously relevant to the
    /// slot.
    RetainRelevant,
}

/// Numerical method used to update a region's trust-region controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionController {
    /// TuRBO-style success and failure counters adapt region length.
    Turbo,
    /// Keep the configured length fixed while retaining relevance and credit.
    FixedRadius,
}

impl Default for RegionController {
    fn default() -> Self {
        Self::Turbo
    }
}

impl Default for RestartMemoryPolicy {
    fn default() -> Self {
        Self::Fresh
    }
}

impl SharingPolicy {
    /// Decide local-model relevance separately from proposal origin and credit.
    pub(super) fn accepts(
        self,
        state: &MultiTrustRegionState,
        region: usize,
        observation: usize,
        x: &ArrayView1<f64>,
        lb: &Array1<f64>,
        ub: &Array1<f64>,
        origins: Option<&[usize]>,
    ) -> bool {
        match self {
            Self::Shared => state.contains(x, lb, ub),
            Self::Independent => origins.is_some_and(|ids| ids[observation] == region),
            Self::NearestCenter => {
                let distance = state.distance_to(region, x);
                (0..state.num_regions)
                    .filter(|&other| other != region && state.active_mask[other])
                    .all(|other| state.distance_to(other, x) >= distance)
            }
        }
    }
}

impl MultiTrustRegionState {
    pub(crate) fn restart_memory(&self) -> RestartMemoryPolicy {
        self.config.restart_memory_policy
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

    use crate::optimizer::{MultiTrustRegionConfig, MultiTrustRegionState};
    use crate::trust_region::TRLengthConfig;

    use super::{AllocationPolicy, RestartCenterPolicy, RestartCenterSelection};

    #[test]
    fn restartpolicy() {
        let centers = array![[0.1, 0.1], [0.8, 0.8]];
        let mut initial = StdRng::seed_from_u64(7);
        let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        config.restart_center_policy = RestartCenterPolicy {
            candidates: 64,
            selection: RestartCenterSelection::First,
        };
        let first =
            MultiTrustRegionState::new(2, config.clone(), Some(&centers.view()), &mut initial)
                .unwrap();
        config.restart_center_policy.selection = RestartCenterSelection::Maximin;
        let maximin =
            MultiTrustRegionState::new(2, config, Some(&centers.view()), &mut initial).unwrap();
        let mut left = StdRng::seed_from_u64(11);
        let mut right = StdRng::seed_from_u64(11);
        let uniform = first.choose_center(0, &mut left).unwrap();
        let distant = maximin.choose_center(0, &mut right).unwrap();
        let distance = |point: &ndarray::Array1<f64>| {
            point
                .iter()
                .zip(centers.row(1))
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
        };
        assert!(distance(&distant) >= distance(&uniform));
        assert_eq!(left.next_u64(), right.next_u64());

        let mut baseline = StdRng::seed_from_u64(13);
        let mut sampled = StdRng::seed_from_u64(13);
        let expected = ndarray::Array1::from_shape_fn(2, |_| baseline.gen_range(0.0..1.0));
        let mut one = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        one.restart_center_policy = RestartCenterPolicy::default();
        let state =
            MultiTrustRegionState::new(2, one, Some(&centers.view()), &mut initial).unwrap();
        assert_eq!(state.choose_center(0, &mut sampled).unwrap(), expected);
        assert_eq!(sampled.next_u64(), baseline.next_u64());
    }

    #[test]
    fn balancedallocation() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut config = MultiTrustRegionConfig::new(3, TRLengthConfig::default());
        config.allocation_policy = AllocationPolicy::Balanced;
        let state = MultiTrustRegionState::new(1, config, None, &mut rng).unwrap();
        let plan = state
            .allocate_with(5, &array![100.0, -5.0, 0.0].view())
            .unwrap();
        assert_eq!(
            plan.iter().map(|batch| batch.region).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            plan.iter().map(|batch| batch.len).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert_eq!(
            plan.iter().map(|batch| batch.start).collect::<Vec<_>>(),
            [0, 2, 4]
        );
    }

    #[test]
    fn balancedhistory() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut config = MultiTrustRegionConfig::new(3, TRLengthConfig::default());
        config.allocation_policy = AllocationPolicy::Balanced;
        let mut state = MultiTrustRegionState::new(1, config, None, &mut rng).unwrap();
        let utility = array![100.0, -5.0, 0.0];
        for expected in [0, 1, 2, 0, 1, 2] {
            let plan = state.allocate_with(1, &utility.view()).unwrap();
            assert_eq!(plan.len(), 1);
            assert_eq!(plan[0].region, expected);
            state.ledger.observe(expected, &[0.0], 0.0).unwrap();
        }
    }
}
