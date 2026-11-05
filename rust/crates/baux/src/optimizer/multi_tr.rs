//! Low-Latency Multi-Trust Region (TuRBO-M) state engine.
//!
//! Maintains $M$ concurrent trust regions in contiguous array memory (SoA
//! layout) for zero-allocation CPU SIMD operations and direct GPU buffer
//! mirroring.

use baux::semantics::optimizer::portfolio::{Allocation, RegionTicket, RegionalLedger};
use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::optimizer_config::AcquisitionConfig;
use crate::trust_region::{TRLengthConfig, TrustRegionError};
mod policy;
mod selection;
pub use policy::{
    AllocationPolicy, RegionController, RestartCenterPolicy, RestartCenterSelection,
    RestartMemoryPolicy,
};

/// Policy for admitting observations into regional models and incumbents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum SharingPolicy {
    /// Any observation may update a containing region's incumbent.
    Shared,
    /// An observation may update the nearest region's incumbent.
    NearestCenter,
    /// Only the generating region may update its incumbent.
    Independent,
}

impl Default for SharingPolicy {
    fn default() -> Self {
        Self::Shared
    }
}

/// Configuration for Multi-Trust Region state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTrustRegionConfig {
    /// Number of concurrent trust regions (M).
    pub num_regions: usize,
    /// Trust region length configuration parameters.
    pub length: TRLengthConfig,
    /// Consecutive successes required before region expansion.
    pub succ_tolerance: i32,
    /// Consecutive failures allowed before region contraction.
    pub fail_tolerance: i32,
    /// Observation sharing policy across regions.
    pub sharing_policy: SharingPolicy,
    /// Candidate-budget allocation policy, independent of observation sharing.
    pub allocation_policy: AllocationPolicy,
    /// Local-model memory policy when a region slot is restarted.
    pub restart_memory_policy: RestartMemoryPolicy,
    /// Candidate search used to replace an automatically retired slot.
    pub restart_center_policy: RestartCenterPolicy,
    /// Numerical regional controller method, independent of acquisition choice.
    pub controller: RegionController,
    /// Per-slot acquisition overrides; `None` inherits the optimizer default.
    pub region_acquisitions: Vec<Option<AcquisitionConfig>>,
    /// UCB exploration weight used to divide a candidate budget.
    pub exploration: f64,
    /// Minimum normalized Hamming distance between selected perturbation seeds.
    pub min_seed_distance: f64,
}

impl MultiTrustRegionConfig {
    pub fn new(num_regions: usize, length: TRLengthConfig) -> Self {
        Self {
            num_regions,
            length,
            succ_tolerance: 3,
            fail_tolerance: 5,
            sharing_policy: SharingPolicy::Shared,
            allocation_policy: AllocationPolicy::UtilityUcb,
            restart_memory_policy: RestartMemoryPolicy::Fresh,
            restart_center_policy: RestartCenterPolicy::default(),
            controller: RegionController::Turbo,
            region_acquisitions: vec![None; num_regions],
            exploration: 1.0,
            min_seed_distance: 0.0,
        }
    }
}

pub use baux::semantics::optimizer::portfolio::RegionBatch;

/// A scored perturbation candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionCandidate {
    pub index: usize,
    pub region: usize,
    pub seed: u64,
    pub score: f64,
}

/// Historical identity of the region instance that proposed an arm.
///
/// A slot may be restarted, so its generation is part of the identity. This
/// records provenance only; observation relevance and progress attribution are
/// separate policy decisions. See `research/multiregion_optimizer.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionProvenance {
    pub slot: usize,
    pub generation: u64,
    pub(crate) ticket: RegionTicket,
}

/// Identity of one generated evaluation, distinct from its region generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvaluationId(pub u64);

/// Outcome of a generated evaluation; cancellation has no measured reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EvaluationStatus {
    Pending,
    Observed(f64),
    Cancelled,
}

/// Identity and completion state for outstanding regional groups.
///
/// A settled slot may append a new group while another slot remains pending.
/// Warm-up remains one ordered batch; overlapping regional work uses IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingProposal {
    /// Process-wide unique evaluation identities, one per ask row.
    pub(crate) ids: Vec<EvaluationId>,
    /// Region instances for a regional batch; `None` identifies warm-up rows.
    pub(crate) arms: Option<Vec<RegionProvenance>>,
    /// Ordered coordinates expected by the legacy complete-batch tell path.
    /// This is a consistency check, not an evaluation identity.
    pub(crate) x: Array2<f64>,
    /// Resolved and unresolved rows. Each region settles after its own rows
    /// have produced observations or been cancelled, independently of other
    /// regions in the batch.
    pub(crate) outcomes: Vec<EvaluationStatus>,
    /// Ordered tell is valid only before another ask has joined this ledger.
    pub(crate) ordered_tell_allowed: bool,
}

/// Multi-Trust Region engine storing $M$ regions in contiguous flat memory.
#[derive(Debug, Clone)]
pub struct MultiTrustRegionState {
    num_regions: usize,
    num_dim: usize,

    /// Contiguous center matrix of shape `[num_regions, num_dim]` in unit
    /// hypercube `[0, 1]^d`.
    pub centers: Array2<f64>,
    /// Contiguous length vector of shape `[num_regions]`.
    pub lengths: Array1<f64>,
    /// Success counters per region.
    pub succ_counts: Vec<i32>,
    /// Failure counters per region.
    pub fail_counts: Vec<i32>,
    /// Active mask per region (false when region needs restart).
    pub active_mask: Vec<bool>,
    ledger: RegionalLedger,

    config: MultiTrustRegionConfig,
}

impl MultiTrustRegionState {
    /// Initialize a new MultiTrustRegionState.
    pub fn new(
        num_dim: usize,
        config: MultiTrustRegionConfig,
        initial_centers: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        let length = config.length;
        if num_dim == 0
            || config.succ_tolerance <= 0
            || config.fail_tolerance <= 0
            || !length.length_min.is_finite()
            || !length.length_init.is_finite()
            || !length.length_max.is_finite()
            || length.length_min <= 0.0
            || length.length_min > length.length_init
            || length.length_init > length.length_max
        {
            return Err(TrustRegionError::InvalidParameter(
                "regions require positive dimensions/tolerances and finite 0 < min <= initial <= max lengths".into(),
            ));
        }
        if config.num_regions == 0 {
            return Err(TrustRegionError::InvalidParameter(
                "num_regions must be > 0".to_string(),
            ));
        }
        if config.restart_center_policy.candidates == 0 {
            return Err(TrustRegionError::InvalidParameter(
                "restart center policy needs at least one candidate".into(),
            ));
        }
        if config.region_acquisitions.len() != config.num_regions {
            return Err(TrustRegionError::InvalidParameter(format!(
                "region_acquisitions has {} entries; expected {}",
                config.region_acquisitions.len(),
                config.num_regions
            )));
        }
        if config.region_acquisitions.iter().flatten().any(|acquisition| {
            matches!(acquisition, AcquisitionConfig::UCB { beta } if !beta.is_finite() || !(*beta as f32).is_finite())
        }) {
            return Err(TrustRegionError::InvalidParameter(
                "regional UCB beta must be finite FP32".into(),
            ));
        }
        if !config.exploration.is_finite() || config.exploration < 0.0 {
            return Err(TrustRegionError::InvalidParameter(
                "exploration must be finite and non-negative".to_string(),
            ));
        }
        if !config.min_seed_distance.is_finite() || !(0.0..=1.0).contains(&config.min_seed_distance)
        {
            return Err(TrustRegionError::InvalidParameter(
                "min_seed_distance must be in [0, 1]".to_string(),
            ));
        }

        let num_regions = config.num_regions;
        let mut centers = Array2::zeros((num_regions, num_dim));

        if let Some(init) = initial_centers {
            if init.nrows() != num_regions || init.ncols() != num_dim {
                return Err(TrustRegionError::InvalidParameter(format!(
                    "initial_centers shape {:?} does not match [{}, {}]",
                    (init.nrows(), init.ncols()),
                    num_regions,
                    num_dim
                )));
            }
            centers.assign(init);
        } else {
            // Uniformly sample centers in unit hypercube [0, 1]^d
            for r in 0..num_regions {
                for d in 0..num_dim {
                    centers[[r, d]] = rand::Rng::gen_range(rng, 0.0..1.0);
                }
            }
        }

        let lengths = Array1::from_elem(num_regions, config.length.length_init);
        let succ_counts = vec![0; num_regions];
        let fail_counts = vec![0; num_regions];
        let active_mask = vec![true; num_regions];
        let allocation = match config.allocation_policy {
            AllocationPolicy::Balanced => Allocation::Balanced,
            AllocationPolicy::UtilityUcb => Allocation::Ucb {
                exploration: config.exploration,
            },
        };
        let ledger = RegionalLedger::new(&vec![f64::NEG_INFINITY; num_regions], allocation)
            .map_err(|error| TrustRegionError::InvalidParameter(error.to_string()))?;

        Ok(Self {
            num_regions,
            num_dim,
            centers,
            lengths,
            succ_counts,
            fail_counts,
            active_mask,
            ledger,
            config,
        })
    }

    pub fn num_regions(&self) -> usize {
        self.num_regions
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn active_count(&self) -> usize {
        self.active_mask.iter().filter(|&&a| a).count()
    }

    pub fn sharing_policy(&self) -> SharingPolicy {
        self.config.sharing_policy
    }

    /// Resolve a slot's acquisition override against the optimizer default.
    pub fn acquisition_for(&self, region: usize, default: AcquisitionConfig) -> AcquisitionConfig {
        self.config
            .region_acquisitions
            .get(region)
            .copied()
            .flatten()
            .unwrap_or(default)
    }

    /// Current lifetime number for a region slot.
    pub fn generation(&self, region: usize) -> Option<u64> {
        self.ledger.generation(region)
    }

    pub(crate) fn is_pending(&self, region: usize) -> Option<bool> {
        self.ledger.is_pending(region)
    }

    pub fn trial_counts(&self) -> Vec<u64> {
        self.ledger.trials()
    }

    pub fn incumbent(&self, region: usize) -> Option<f64> {
        self.ledger.best(region)
    }

    pub fn incumbents(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.num_regions, |slot| {
            self.ledger
                .best(slot)
                .expect("regional ledger matches numerical state")
        })
    }

    pub(crate) fn issue(&mut self, region: usize) -> Result<RegionTicket, TrustRegionError> {
        self.ledger
            .issue(region)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
    }

    pub(crate) fn validate(&self, ticket: RegionTicket) -> Result<(), TrustRegionError> {
        self.ledger
            .validate(ticket)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
    }

    pub(crate) fn finish(&mut self, ticket: RegionTicket) -> Result<(), TrustRegionError> {
        self.ledger
            .finish(ticket)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
    }

    pub(crate) fn retire(&mut self, ticket: RegionTicket) -> Result<(), TrustRegionError> {
        self.ledger
            .retire(ticket)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
    }

    /// Update trust regions with newly observed batch points `x_batch` (shape
    /// [N, d]) and `y_batch` (shape [N]).
    pub fn tell_update(
        &mut self,
        x_batch: &ArrayView2<f64>,
        y_batch: &ArrayView1<f64>,
    ) -> Result<(), TrustRegionError> {
        self.tell_scaled(x_batch, y_batch, None, None)
    }

    /// Update regions, optionally preserving each observation's generating
    /// region.
    pub fn tell(
        &mut self,
        x_batch: &ArrayView2<f64>,
        y_batch: &ArrayView1<f64>,
        regions: Option<&[usize]>,
    ) -> Result<(), TrustRegionError> {
        self.tell_scaled(x_batch, y_batch, regions, None)
    }

    /// Update regions with optional generating region IDs and surrogate
    /// lengthscales.
    pub fn tell_scaled(
        &mut self,
        x_batch: &ArrayView2<f64>,
        y_batch: &ArrayView1<f64>,
        regions: Option<&[usize]>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> Result<(), TrustRegionError> {
        let n = x_batch.nrows();
        if n == 0 {
            return Ok(());
        }
        if x_batch.ncols() != self.num_dim || y_batch.len() != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "Batch shapes mismatched: x {:?}, y len {}",
                (x_batch.nrows(), x_batch.ncols()),
                y_batch.len()
            )));
        }
        self.validate_regions(regions, n)?;

        for r in 0..self.num_regions {
            if !self.active_mask[r] {
                continue;
            }

            let mut region_improved = false;
            let mut credit_improved = false;
            let current_best = self
                .ledger
                .best(r)
                .expect("regional ledger matches numerical state");
            let mut credited_rewards = Vec::new();
            let mut best_y_in_batch = f64::NEG_INFINITY;
            let mut best_x_idx: Option<usize> = None;

            let (lb, ub) = self.compute_bounds(r, lengthscales);

            for i in 0..n {
                let x_i = x_batch.row(i);
                let y_i = y_batch[i];

                let inside = self.accepts(r, i, &x_i, &lb, &ub, regions);

                if regions.is_some_and(|ids| ids[i] == r) {
                    credited_rewards.push(y_i);
                    credit_improved |= inside && y_i > current_best;
                }
                if inside && y_i > current_best && y_i > best_y_in_batch {
                    best_y_in_batch = y_i;
                    best_x_idx = Some(i);
                    region_improved = true;
                }
            }

            if region_improved {
                if let Some(idx) = best_x_idx {
                    self.ledger
                        .set_best(r, y_batch[idx])
                        .map_err(|error| TrustRegionError::InvalidState(error.to_string()))?;
                    let x_best = x_batch.row(idx);
                    for d in 0..self.num_dim {
                        self.centers[[r, d]] = x_best[d];
                    }
                }
            }
            if credited_rewards.is_empty() {
                continue;
            }
            let incumbent = self
                .ledger
                .best(r)
                .expect("regional ledger matches numerical state");
            self.ledger
                .observe(r, &credited_rewards, incumbent)
                .map_err(|error| TrustRegionError::InvalidState(error.to_string()))?;
            self.update_controller(r, credit_improved);
        }

        Ok(())
    }

    fn validate_regions(
        &self,
        regions: Option<&[usize]>,
        n: usize,
    ) -> Result<(), TrustRegionError> {
        if let Some(ids) = regions {
            if ids.len() != n || ids.iter().any(|&r| r >= self.num_regions) {
                return Err(TrustRegionError::InvalidParameter(
                    "generating regions do not match the observation batch".to_string(),
                ));
            }
        } else if self.config.sharing_policy == SharingPolicy::Independent {
            return Err(TrustRegionError::InvalidParameter(
                "independent sharing requires generating regions".to_string(),
            ));
        }
        Ok(())
    }

    fn update_controller(&mut self, region: usize, improved: bool) {
        // BAUX's fixed-length baseline keeps observation relevance and origin
        // credit but deliberately disables TuRBO length adaptation.
        if self.config.controller == RegionController::FixedRadius {
            return;
        }
        if improved {
            self.succ_counts[region] += 1;
            self.fail_counts[region] = 0;
            if self.succ_counts[region] >= self.config.succ_tolerance {
                self.lengths[region] =
                    (2.0 * self.lengths[region]).min(self.config.length.length_max);
                self.succ_counts[region] = 0;
            }
        } else {
            self.succ_counts[region] = 0;
            self.fail_counts[region] += 1;
            if self.fail_counts[region] >= self.config.fail_tolerance {
                self.lengths[region] *= 0.5;
                self.fail_counts[region] = 0;
                if self.lengths[region] < self.config.length.length_min {
                    self.active_mask[region] = false;
                }
            }
        }
    }

    /// Restart a inactive region `r` with a new center location.
    pub fn restart_region(
        &mut self,
        r: usize,
        new_center: &ArrayView1<f64>,
    ) -> Result<(), TrustRegionError> {
        if r >= self.num_regions {
            return Err(TrustRegionError::InvalidParameter(format!(
                "region_idx {} out of bounds {}",
                r, self.num_regions
            )));
        }
        if new_center.len() != self.num_dim {
            return Err(TrustRegionError::InvalidParameter(format!(
                "new_center len {} != num_dim {}",
                new_center.len(),
                self.num_dim
            )));
        }

        self.ledger
            .restart(r, f64::NEG_INFINITY)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))?;

        for d in 0..self.num_dim {
            self.centers[[r, d]] = new_center[d].clamp(0.0, 1.0);
        }
        self.lengths[r] = self.config.length.length_init;
        self.succ_counts[r] = 0;
        self.fail_counts[r] = 0;
        self.active_mask[r] = true;

        Ok(())
    }

    pub(crate) fn accepts(
        &self,
        region: usize,
        observation: usize,
        x: &ArrayView1<f64>,
        lb: &Array1<f64>,
        ub: &Array1<f64>,
        regions: Option<&[usize]>,
    ) -> bool {
        self.config
            .sharing_policy
            .accepts(self, region, observation, x, lb, ub, regions)
    }

    fn contains(&self, x: &ArrayView1<f64>, lb: &Array1<f64>, ub: &Array1<f64>) -> bool {
        (0..self.num_dim).all(|d| x[d] >= lb[d] && x[d] <= ub[d])
    }

    fn distance_to(&self, region: usize, x: &ArrayView1<f64>) -> f64 {
        let center = self.centers.row(region);
        (0..self.num_dim).map(|d| (x[d] - center[d]).powi(2)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer_config::AcquisitionConfig;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_001() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = MultiTrustRegionConfig::new(4, TRLengthConfig::default());
        let tr = MultiTrustRegionState::new(3, cfg, None, &mut rng).unwrap();

        assert_eq!(tr.num_regions(), 4);
        assert_eq!(tr.num_dim(), 3);
        assert_eq!(tr.active_count(), 4);
        assert_eq!(tr.centers.shape(), &[4, 3]);
    }

    #[test]
    fn acquisitionoverride() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        config.region_acquisitions[1] = Some(AcquisitionConfig::Random);
        let state = MultiTrustRegionState::new(1, config.clone(), None, &mut rng).unwrap();
        let global = AcquisitionConfig::UCB { beta: 1.25 };
        assert_eq!(state.acquisition_for(0, global), global);
        assert_eq!(state.acquisition_for(1, global), AcquisitionConfig::Random);

        config.region_acquisitions.pop();
        assert!(MultiTrustRegionState::new(1, config, None, &mut rng).is_err());

        let mut config = MultiTrustRegionConfig::new(1, TRLengthConfig::default());
        config.region_acquisitions[0] = Some(AcquisitionConfig::UCB {
            beta: f64::INFINITY,
        });
        assert!(MultiTrustRegionState::new(1, config, None, &mut rng).is_err());
    }

    #[test]
    fn fixedlength() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut config = MultiTrustRegionConfig::new(
            1,
            TRLengthConfig {
                length_min: 0.1,
                length_init: 0.25,
                length_max: 0.5,
            },
        );
        config.controller = RegionController::FixedRadius;
        config.succ_tolerance = 1;
        config.fail_tolerance = 1;
        let center = array![[0.5]];
        let mut state =
            MultiTrustRegionState::new(1, config, Some(&center.view()), &mut rng).unwrap();
        state
            .tell(&array![[0.5]].view(), &array![4.0].view(), Some(&[0]))
            .unwrap();
        assert_eq!(state.incumbent(0), Some(4.0));
        assert_eq!(state.trial_counts(), [1]);
        assert_eq!(state.succ_counts, [0]);
        assert_eq!(state.fail_counts, [0]);
        assert_eq!(state.lengths[0], 0.25);
        assert!(state.active_mask[0]);
    }

    #[test]
    fn test_002() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cfg = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        cfg.succ_tolerance = 2;
        cfg.fail_tolerance = 2;

        let init_centers = array![[0.5, 0.5], [0.1, 0.1]];
        let mut tr =
            MultiTrustRegionState::new(2, cfg, Some(&init_centers.view()), &mut rng).unwrap();

        let initial_len0 = tr.lengths[0];

        // Tell improvement near center 0
        let x_batch = array![[0.51, 0.51]];
        let y_batch = array![10.0];
        tr.tell(&x_batch.view(), &y_batch.view(), Some(&[0]))
            .unwrap();
        assert_eq!(tr.succ_counts[0], 1);

        let y_batch2 = array![15.0];
        tr.tell(&x_batch.view(), &y_batch2.view(), Some(&[0]))
            .unwrap();
        assert!(tr.lengths[0] > initial_len0, "Region 0 should expand");
    }

    #[test]
    fn sharing_credit() {
        let mut rng = StdRng::seed_from_u64(8);
        let centers = array![[0.5], [0.5]];
        let mut state = MultiTrustRegionState::new(
            1,
            MultiTrustRegionConfig::new(2, Default::default()),
            Some(&centers.view()),
            &mut rng,
        )
        .unwrap();
        state
            .tell(&array![[0.5]].view(), &array![10.0].view(), Some(&[0]))
            .unwrap();
        assert_eq!(state.incumbents().to_vec(), vec![10.0, 10.0]);
        assert_eq!(state.trial_counts(), vec![1, 0]);
        assert_eq!(state.succ_counts, vec![1, 0]);
        assert_eq!(state.fail_counts, vec![0, 0]);

        state
            .tell(
                &array![[0.5], [0.5]].view(),
                &array![9.0, 20.0].view(),
                Some(&[0, 1]),
            )
            .unwrap();
        assert_eq!(state.incumbents().to_vec(), vec![20.0, 20.0]);
        assert_eq!(state.trial_counts(), vec![2, 1]);
        assert_eq!(state.fail_counts, vec![1, 0]);
        assert_eq!(state.succ_counts, vec![0, 1]);

        state
            .tell_update(&array![[0.5]].view(), &array![30.0].view())
            .unwrap();
        assert_eq!(state.incumbents().to_vec(), vec![30.0, 30.0]);
        assert_eq!(state.trial_counts(), vec![2, 1]);
        assert_eq!(state.fail_counts, vec![1, 0]);
        assert_eq!(state.succ_counts, vec![0, 1]);
    }

    #[test]
    fn allocation_trajectory() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tr = MultiTrustRegionState::new(
            2,
            MultiTrustRegionConfig::new(2, Default::default()),
            None,
            &mut rng,
        )
        .unwrap();
        let x = array![[0.1, 0.1], [0.9, 0.9]];
        tr.tell(&x.view(), &array![10.0, 1.0].view(), Some(&[0, 1]))
            .unwrap();

        let plan = tr.allocate(8).unwrap();
        assert!(plan[0].len > plan[1].len);
        assert_eq!(plan.iter().map(|batch| batch.len).sum::<usize>(), 8);
    }

    #[test]
    fn allocation_regions() {
        let mut rng = StdRng::seed_from_u64(42);
        let tr = MultiTrustRegionState::new(
            2,
            MultiTrustRegionConfig::new(2, Default::default()),
            None,
            &mut rng,
        )
        .unwrap();

        let plan = tr.allocate(8).unwrap();
        assert_eq!(
            plan.iter().map(|batch| batch.len).collect::<Vec<_>>(),
            [4, 4]
        );
    }

    #[test]
    fn allocation_utility() {
        let mut rng = StdRng::seed_from_u64(42);
        let tr = MultiTrustRegionState::new(
            2,
            MultiTrustRegionConfig::new(2, Default::default()),
            None,
            &mut rng,
        )
        .unwrap();

        let plan = tr.allocate_with(6, &array![0.0, 10.0].view()).unwrap();
        assert!(plan[1].len > plan[0].len);
        assert!(tr.allocate_with(6, &array![f64::NAN].view()).is_err());
    }

    #[test]
    fn allocation_budgets() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tr = MultiTrustRegionState::new(
            2,
            MultiTrustRegionConfig::new(4, Default::default()),
            None,
            &mut rng,
        )
        .unwrap();
        for (slot, reward) in [1.0, 4.0, 2.0, 3.0].into_iter().enumerate() {
            tr.ledger.observe(slot, &[reward], reward).unwrap();
        }

        let plan = tr.allocate(2).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.iter().map(|batch| batch.len).sum::<usize>(), 2);
        assert!(plan.iter().all(|batch| batch.len == 1));
        assert!(plan[0].region != plan[1].region);
    }

    #[test]
    fn selection_diverse() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cfg = MultiTrustRegionConfig::new(2, Default::default());
        cfg.min_seed_distance = 0.25;
        let tr = MultiTrustRegionState::new(2, cfg, None, &mut rng).unwrap();
        let candidates = [
            RegionCandidate {
                index: 0,
                region: 0,
                seed: 0,
                score: 9.0,
            },
            RegionCandidate {
                index: 1,
                region: 0,
                seed: 1,
                score: 8.0,
            },
            RegionCandidate {
                index: 2,
                region: 1,
                seed: u64::MAX,
                score: 7.0,
            },
        ];

        let selected = tr.select(&candidates, 2).unwrap();
        assert_eq!(selected.iter().map(|c| c.index).collect::<Vec<_>>(), [0, 2]);
    }

    #[test]
    fn independent_region() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cfg = MultiTrustRegionConfig::new(2, Default::default());
        cfg.sharing_policy = SharingPolicy::Independent;
        let centers = array![[0.5], [0.5]];
        let mut tr = MultiTrustRegionState::new(1, cfg, Some(&centers.view()), &mut rng).unwrap();

        assert!(
            tr.tell_update(&array![[0.5]].view(), &array![3.0].view())
                .is_err()
        );
        tr.tell(&array![[0.5]].view(), &array![3.0].view(), Some(&[1]))
            .unwrap();
        assert_eq!(tr.incumbent(0), Some(f64::NEG_INFINITY));
        assert_eq!(tr.incumbent(1), Some(3.0));
        assert_eq!(tr.trial_counts(), [0, 1]);
        tr.tell(&array![[0.6]].view(), &array![5.0].view(), Some(&[1]))
            .unwrap();
        assert_eq!(tr.variance(1), Some(2.0));
        assert_eq!(tr.variance(2), None);
    }

    #[test]
    fn nearest_explicit() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cfg = MultiTrustRegionConfig::new(2, Default::default());
        cfg.sharing_policy = SharingPolicy::NearestCenter;
        cfg.min_seed_distance = 0.5;
        let centers = array![[0.0], [1.0]];
        let tr = MultiTrustRegionState::new(1, cfg, Some(&centers.view()), &mut rng).unwrap();
        let x = array![0.1];
        let lb = array![0.0];
        let ub = array![1.0];

        assert!(tr.accepts(0, 0, &x.view(), &lb, &ub, None));
        assert!(!tr.accepts(1, 0, &x.view(), &lb, &ub, None));
        assert!(tr.distance_to(0, &x.view()) < tr.distance_to(1, &x.view()));
        assert!(tr.seed_distance(0, u64::MAX));
        assert!(!tr.seed_distance(0, 1));
    }

    #[test]
    fn generation_restart() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut tr = MultiTrustRegionState::new(
            2,
            MultiTrustRegionConfig::new(2, Default::default()),
            None,
            &mut rng,
        )
        .unwrap();

        assert_eq!(tr.generation(0), Some(0));
        assert_eq!(tr.generation(1), Some(0));
        assert_eq!(tr.generation(2), None);

        tr.restart_region(0, &array![0.25, 0.75].view()).unwrap();
        assert_eq!(tr.generation(0), Some(1));
        assert_eq!(tr.generation(1), Some(0));

        tr.restart_region(0, &array![0.75, 0.25].view()).unwrap();
        assert_eq!(tr.generation(0), Some(2));
    }
}
