//! Regional lifecycle and scheduling shared by optimizer adapters.
//!
//! The ledger owns identities, outstanding-group capacity, reward statistics,
//! allocation, and restart generations. Host and resident adapters retain
//! their own numerical state and storage. This keeps execution-specific arrays
//! and device buffers out of the shared interface.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

static OWNERS: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionId {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionTicket {
    owner: u64,
    sequence: u64,
    pub region: RegionId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionBatch {
    pub region: usize,
    pub start: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Allocation {
    Balanced,
    Ucb { exploration: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionalError {
    CounterExhausted,
    InvalidAllocation,
    InvalidOutcome,
    InvalidRegion,
    NoAvailableRegion,
    RegionBusy,
    StaleTicket,
}

impl fmt::Display for RegionalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::CounterExhausted => "regional identity or trial counter exhausted",
            Self::InvalidAllocation => "regional allocation inputs are invalid",
            Self::InvalidOutcome => "regional outcomes must be finite",
            Self::InvalidRegion => "unknown region",
            Self::NoAvailableRegion => "all regional slots have outstanding evaluations",
            Self::RegionBusy => "region already has an outstanding group",
            Self::StaleTicket => "foreign, stale, or completed regional ticket",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RegionalError {}

#[derive(Debug, Clone)]
struct Region {
    generation: u64,
    pending: Option<RegionTicket>,
    trials: u64,
    mean: f64,
    m2: f64,
    best: f64,
}

#[derive(Debug, Clone)]
pub struct RegionalLedger {
    owner: u64,
    next_ticket: u64,
    regions: Vec<Region>,
    allocation: Allocation,
}

impl RegionalLedger {
    pub fn new(initial_best: &[f64], allocation: Allocation) -> Result<Self, RegionalError> {
        if initial_best.is_empty()
            || initial_best
                .iter()
                .any(|value| value.is_nan() || *value == f64::INFINITY)
        {
            return Err(RegionalError::InvalidOutcome);
        }
        if matches!(allocation, Allocation::Ucb { exploration } if !exploration.is_finite() || exploration < 0.0)
        {
            return Err(RegionalError::InvalidAllocation);
        }
        let owner = OWNERS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| RegionalError::CounterExhausted)?;
        Ok(Self {
            owner,
            next_ticket: 1,
            regions: initial_best
                .iter()
                .map(|&best| Region {
                    generation: 0,
                    pending: None,
                    trials: 0,
                    mean: 0.0,
                    m2: 0.0,
                    best,
                })
                .collect(),
            allocation,
        })
    }

    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn generation(&self, slot: usize) -> Option<u64> {
        self.regions.get(slot).map(|region| region.generation)
    }

    pub fn trials(&self) -> Vec<u64> {
        self.regions.iter().map(|region| region.trials).collect()
    }

    pub fn trial_count(&self, slot: usize) -> Option<u64> {
        self.regions.get(slot).map(|region| region.trials)
    }

    pub fn mean(&self, slot: usize) -> Option<f64> {
        self.regions.get(slot).map(|region| region.mean)
    }

    pub fn variance(&self, slot: usize) -> Option<f64> {
        let region = self.regions.get(slot)?;
        (region.trials > 1).then(|| region.m2 / (region.trials - 1) as f64)
    }

    pub fn best(&self, slot: usize) -> Option<f64> {
        self.regions.get(slot).map(|region| region.best)
    }

    pub fn best_region(&self) -> (usize, f64) {
        self.regions
            .iter()
            .enumerate()
            .max_by(|(left_slot, left), (right_slot, right)| {
                left.best
                    .total_cmp(&right.best)
                    .then_with(|| right_slot.cmp(left_slot))
            })
            .map(|(slot, region)| (slot, region.best))
            .expect("regional ledger is nonempty")
    }

    pub fn is_pending(&self, slot: usize) -> Option<bool> {
        self.regions
            .get(slot)
            .map(|region| region.pending.is_some())
    }

    pub fn allocate(
        &self,
        budget: usize,
        eligible: &[bool],
    ) -> Result<Vec<RegionBatch>, RegionalError> {
        let utility = self
            .regions
            .iter()
            .map(|region| region.mean)
            .collect::<Vec<_>>();
        self.allocate_with(budget, eligible, &utility)
    }

    pub fn allocate_with(
        &self,
        budget: usize,
        eligible: &[bool],
        utility: &[f64],
    ) -> Result<Vec<RegionBatch>, RegionalError> {
        if budget == 0
            || eligible.len() != self.len()
            || utility.len() != self.len()
            || utility.iter().any(|value| !value.is_finite())
        {
            return Err(RegionalError::InvalidAllocation);
        }
        let available = self
            .regions
            .iter()
            .enumerate()
            .filter_map(|(slot, region)| {
                (eligible[slot] && region.pending.is_none()).then_some(slot)
            })
            .collect::<Vec<_>>();
        if available.is_empty() {
            return Err(RegionalError::NoAvailableRegion);
        }
        match self.allocation {
            Allocation::Balanced => self.allocate_balanced(budget, available),
            Allocation::Ucb { exploration } => {
                self.allocate_ucb(budget, available, utility, exploration)
            }
        }
    }

    pub fn issue(&mut self, slot: usize) -> Result<RegionTicket, RegionalError> {
        let region = self
            .regions
            .get_mut(slot)
            .ok_or(RegionalError::InvalidRegion)?;
        if region.pending.is_some() {
            return Err(RegionalError::RegionBusy);
        }
        let sequence = self.next_ticket;
        self.next_ticket = sequence
            .checked_add(1)
            .ok_or(RegionalError::CounterExhausted)?;
        let ticket = RegionTicket {
            owner: self.owner,
            sequence,
            region: RegionId {
                slot,
                generation: region.generation,
            },
        };
        region.pending = Some(ticket);
        Ok(ticket)
    }

    pub fn validate(&self, ticket: RegionTicket) -> Result<(), RegionalError> {
        if ticket.owner != self.owner
            || self
                .regions
                .get(ticket.region.slot)
                .and_then(|region| region.pending)
                != Some(ticket)
        {
            return Err(RegionalError::StaleTicket);
        }
        Ok(())
    }

    pub fn complete(
        &mut self,
        ticket: RegionTicket,
        rewards: &[f64],
        incumbent: f64,
    ) -> Result<(), RegionalError> {
        self.validate(ticket)?;
        self.observe(ticket.region.slot, rewards, incumbent)?;
        self.finish(ticket)
    }

    pub fn observe(
        &mut self,
        slot: usize,
        rewards: &[f64],
        incumbent: f64,
    ) -> Result<(), RegionalError> {
        if rewards.iter().any(|value| !value.is_finite())
            || incumbent.is_nan()
            || incumbent == f64::INFINITY
        {
            return Err(RegionalError::InvalidOutcome);
        }
        let region = self
            .regions
            .get_mut(slot)
            .ok_or(RegionalError::InvalidRegion)?;
        let added = u64::try_from(rewards.len()).map_err(|_| RegionalError::CounterExhausted)?;
        region
            .trials
            .checked_add(added)
            .ok_or(RegionalError::CounterExhausted)?;
        for &reward in rewards {
            region.trials += 1;
            let count = region.trials as f64;
            let delta = reward - region.mean;
            region.mean += delta / count;
            region.m2 += delta * (reward - region.mean);
        }
        region.best = incumbent;
        Ok(())
    }

    pub fn set_best(&mut self, slot: usize, incumbent: f64) -> Result<(), RegionalError> {
        if incumbent.is_nan() || incumbent == f64::INFINITY {
            return Err(RegionalError::InvalidOutcome);
        }
        self.regions
            .get_mut(slot)
            .ok_or(RegionalError::InvalidRegion)?
            .best = incumbent;
        Ok(())
    }

    pub fn finish(&mut self, ticket: RegionTicket) -> Result<(), RegionalError> {
        self.validate(ticket)?;
        self.regions[ticket.region.slot].pending = None;
        Ok(())
    }

    pub fn retire(&mut self, ticket: RegionTicket) -> Result<(), RegionalError> {
        self.finish(ticket)
    }

    pub fn restart(
        &mut self,
        slot: usize,
        incumbent: f64,
    ) -> Result<Option<RegionTicket>, RegionalError> {
        if incumbent.is_nan() || incumbent == f64::INFINITY {
            return Err(RegionalError::InvalidOutcome);
        }
        let region = self
            .regions
            .get_mut(slot)
            .ok_or(RegionalError::InvalidRegion)?;
        let generation = region
            .generation
            .checked_add(1)
            .ok_or(RegionalError::CounterExhausted)?;
        let abandoned = region.pending;
        *region = Region {
            generation,
            pending: None,
            trials: 0,
            mean: 0.0,
            m2: 0.0,
            best: incumbent,
        };
        Ok(abandoned)
    }

    fn allocate_balanced(
        &self,
        budget: usize,
        available: Vec<usize>,
    ) -> Result<Vec<RegionBatch>, RegionalError> {
        let mut counts = vec![0usize; self.len()];
        for _ in 0..budget {
            let slot = available
                .iter()
                .copied()
                .min_by_key(|&slot| {
                    (
                        u128::from(self.regions[slot].trials) + counts[slot] as u128,
                        slot,
                    )
                })
                .ok_or(RegionalError::NoAvailableRegion)?;
            counts[slot] += 1;
        }
        Ok(Self::batches(available, &counts))
    }

    fn allocate_ucb(
        &self,
        budget: usize,
        mut available: Vec<usize>,
        utility: &[f64],
        exploration: f64,
    ) -> Result<Vec<RegionBatch>, RegionalError> {
        let total = self
            .regions
            .iter()
            .map(|region| region.trials)
            .sum::<u64>()
            .max(1) as f64;
        if budget < available.len() {
            available.sort_by(|&left, &right| {
                self.priority(right, 0, total, utility[right], exploration)
                    .total_cmp(&self.priority(left, 0, total, utility[left], exploration))
                    .then_with(|| left.cmp(&right))
            });
            let mut counts = vec![0usize; self.len()];
            for &slot in available.iter().take(budget) {
                counts[slot] = 1;
            }
            return Ok(Self::batches(available, &counts));
        }
        let mut counts = vec![0usize; self.len()];
        for &slot in &available {
            counts[slot] = 1;
        }
        for _ in available.len()..budget {
            let slot = available
                .iter()
                .copied()
                .max_by(|&left, &right| {
                    self.priority(left, counts[left], total, utility[left], exploration)
                        .total_cmp(&self.priority(
                            right,
                            counts[right],
                            total,
                            utility[right],
                            exploration,
                        ))
                        .then_with(|| right.cmp(&left))
                })
                .ok_or(RegionalError::NoAvailableRegion)?;
            counts[slot] += 1;
        }
        Ok(Self::batches(available, &counts))
    }

    fn batches(available: Vec<usize>, counts: &[usize]) -> Vec<RegionBatch> {
        let mut start = 0;
        available
            .into_iter()
            .filter_map(|region| {
                let len = counts[region];
                if len == 0 {
                    return None;
                }
                let batch = RegionBatch { region, start, len };
                start += len;
                Some(batch)
            })
            .collect()
    }

    fn priority(
        &self,
        slot: usize,
        queued: usize,
        total: f64,
        utility: f64,
        exploration: f64,
    ) -> f64 {
        if exploration == 0.0 {
            return utility;
        }
        let trials = self.regions[slot].trials as f64;
        let effective = (trials + queued as f64).max(1e-6);
        utility + exploration * ((total + self.len() as f64 + 1.0).ln() / effective).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle() {
        let mut ledger = RegionalLedger::new(&[f64::NEG_INFINITY], Allocation::Balanced).unwrap();
        let ticket = ledger.issue(0).unwrap();
        ledger.complete(ticket, &[1.0, 3.0], 3.0).unwrap();
        assert_eq!(ledger.trials(), [2]);
        assert_eq!(ledger.mean(0), Some(2.0));
        assert_eq!(ledger.variance(0), Some(2.0));
        assert!(ledger.validate(ticket).is_err());
    }

    #[test]
    fn allocation() {
        let mut ledger = RegionalLedger::new(&[0.0; 3], Allocation::Balanced).unwrap();
        ledger.observe(0, &[1.0], 1.0).unwrap();
        let busy = ledger.issue(1).unwrap();
        let batches = ledger.allocate(4, &[true; 3]).unwrap();
        assert_eq!(
            batches,
            [
                RegionBatch {
                    region: 0,
                    start: 0,
                    len: 2,
                },
                RegionBatch {
                    region: 2,
                    start: 2,
                    len: 2,
                },
            ]
        );
        ledger.retire(busy).unwrap();
    }

    #[test]
    fn restart_pending() {
        let mut ledger = RegionalLedger::new(&[1.0], Allocation::Balanced).unwrap();
        let ticket = ledger.issue(0).unwrap();
        assert_eq!(ledger.restart(0, 3.0).unwrap(), Some(ticket));
        assert_eq!(ledger.generation(0), Some(1));
        assert_eq!(ledger.best_region(), (0, 3.0));
        assert!(ledger.validate(ticket).is_err());
        assert_eq!(ledger.trials(), [0]);
    }
}
