//! Regional scheduling and provenance over existing optimizer transactions.
//!
//! BAUX engineering policy: regions own their engines; scheduling never changes
//! an observation's origin. This is not TuRBO-m's joint Thompson allocation.
//! Multi-start replaces an idle engine and increments its lifetime. CPU and
//! resident adapters share this ledger without sharing parameter storage.

use super::RoundId;
use std::sync::atomic::{AtomicU64, Ordering};

static OWNERS: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionId {
    pub slot: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionTicket {
    owner: u64,
    pub region: RegionId,
    pub round: RoundId,
}

#[derive(Debug, Clone, Copy)]
pub enum Allocation {
    /// Balanced independent starts; ties use stable slot order.
    Balanced,
    /// UCB on observed regional rewards; exploration must be nonnegative.
    Ucb { exploration: f64 },
}

struct Region<S> {
    engine: S,
    generation: u64,
    pending: Option<RegionTicket>,
    trials: u64,
    mean: f64,
    best: f64,
}

pub struct Portfolio<S> {
    owner: u64,
    regions: Vec<Region<S>>,
    allocation: Allocation,
}

impl<S> Portfolio<S> {
    pub fn new(starts: Vec<(S, f64)>, allocation: Allocation) -> Result<Self, String> {
        if starts.is_empty() || starts.iter().any(|(_, value)| !value.is_finite()) {
            return Err("portfolio needs finite evaluated starting states".into());
        }
        if let Allocation::Ucb { exploration } = allocation {
            if !exploration.is_finite() || exploration < 0.0 {
                return Err("portfolio exploration must be finite and nonnegative".into());
            }
        }
        let owner = OWNERS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "portfolio owner space exhausted")?;
        Ok(Self {
            owner,
            allocation,
            regions: starts
                .into_iter()
                .map(|(engine, best)| Region {
                    engine,
                    best,
                    generation: 0,
                    pending: None,
                    trials: 0,
                    mean: 0.0,
                })
                .collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.regions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn engine(&self, slot: usize) -> Result<&S, String> {
        self.regions
            .get(slot)
            .map(|r| &r.engine)
            .ok_or("unknown region".into())
    }

    pub fn engine_mut(&mut self, slot: usize) -> Result<&mut S, String> {
        self.regions
            .get_mut(slot)
            .map(|r| &mut r.engine)
            .ok_or("unknown region".into())
    }

    pub fn generation(&self, slot: usize) -> Result<u64, String> {
        self.regions
            .get(slot)
            .map(|r| r.generation)
            .ok_or("unknown region".into())
    }

    pub fn trials(&self) -> Vec<u64> {
        self.regions.iter().map(|r| r.trials).collect()
    }

    /// Best current region incumbent. Replaced trajectories are not retained.
    pub fn best(&self) -> (usize, f64) {
        self.regions
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.best.total_cmp(&y.best).then_with(|| b.cmp(a)))
            .map(|(slot, r)| (slot, r.best))
            .expect("nonempty portfolio")
    }

    pub fn choose(&self) -> Result<usize, String> {
        let total = self.regions.iter().map(|r| r.trials as f64).sum::<f64>();
        let priority = |r: &Region<S>| match self.allocation {
            Allocation::Balanced => -(r.trials as f64),
            Allocation::Ucb { exploration } if r.trials > 0 => {
                r.mean + exploration * ((total + 1.0).ln() / r.trials as f64).sqrt()
            }
            Allocation::Ucb { .. } => f64::INFINITY,
        };
        self.regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.pending.is_none())
            .max_by(|(a, x), (b, y)| priority(x).total_cmp(&priority(y)).then_with(|| b.cmp(a)))
            .map(|(slot, _)| slot)
            .ok_or("all regions have outstanding proposals".into())
    }

    pub fn issue(&mut self, slot: usize, round: RoundId) -> Result<RegionTicket, String> {
        let region = self.regions.get_mut(slot).ok_or("unknown region")?;
        if region.pending.is_some() {
            return Err("region already has a pending proposal".into());
        }
        if region.trials == u64::MAX {
            return Err("region trial counter exhausted".into());
        }
        let ticket = RegionTicket {
            owner: self.owner,
            region: RegionId {
                slot,
                generation: region.generation,
            },
            round,
        };
        region.pending = Some(ticket);
        Ok(ticket)
    }

    pub fn validate(&self, ticket: RegionTicket) -> Result<(), String> {
        if ticket.owner != self.owner
            || self.regions.get(ticket.region.slot).and_then(|r| r.pending) != Some(ticket)
        {
            return Err("foreign, stale, or completed regional proposal".into());
        }
        Ok(())
    }

    pub fn complete(
        &mut self,
        ticket: RegionTicket,
        reward: f64,
        incumbent: f64,
    ) -> Result<(), String> {
        self.validate(ticket)?;
        if !reward.is_finite() || !incumbent.is_finite() {
            return Err("regional outcome must be finite".into());
        }
        let region = &mut self.regions[ticket.region.slot];
        region.trials += 1;
        region.mean += (reward - region.mean) / region.trials as f64;
        region.best = incumbent;
        region.pending = None;
        Ok(())
    }

    /// Retire an unreplayable proposal after its engine reports a partial
    /// mutation. No reward or progress credit is assigned to this ticket.
    pub fn retire(&mut self, ticket: RegionTicket) -> Result<(), String> {
        self.validate(ticket)?;
        self.regions[ticket.region.slot].pending = None;
        Ok(())
    }

    /// Replace a region while explicitly abandoning its outstanding ticket.
    /// The ticket becomes stale because the new engine has a new generation.
    pub fn restart_abandoning(
        &mut self,
        ticket: RegionTicket,
        engine: S,
        value: f64,
    ) -> Result<S, String> {
        self.validate(ticket)?;
        if !value.is_finite() {
            return Err("restart requires a finite initial reward".into());
        }
        let region = &mut self.regions[ticket.region.slot];
        let generation = region
            .generation
            .checked_add(1)
            .ok_or("region generation exhausted")?;
        let previous = std::mem::replace(
            region,
            Region {
                engine,
                best: value,
                generation,
                pending: None,
                trials: 0,
                mean: 0.0,
            },
        );
        Ok(previous.engine)
    }

    /// Install a genuinely new starting state. Outstanding work must finish.
    pub fn restart(&mut self, slot: usize, engine: S, value: f64) -> Result<S, String> {
        let region = self.regions.get_mut(slot).ok_or("unknown region")?;
        if region.pending.is_some() || !value.is_finite() {
            return Err("restart requires an idle region and finite initial reward".into());
        }
        let generation = region
            .generation
            .checked_add(1)
            .ok_or("region generation exhausted")?;
        let previous = std::mem::replace(
            region,
            Region {
                engine,
                best: value,
                generation,
                pending: None,
                trials: 0,
                mean: 0.0,
            },
        );
        Ok(previous.engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_pending() {
        let mut portfolio = Portfolio::new(vec![("old", 1.0)], Allocation::Balanced).unwrap();
        let ticket = portfolio
            .issue(
                0,
                RoundId {
                    owner: 4,
                    proposal: 9,
                    incumbent: 2,
                },
            )
            .unwrap();
        let old = portfolio.restart_abandoning(ticket, "new", 3.0).unwrap();
        assert_eq!(old, "old");
        assert_eq!(portfolio.generation(0).unwrap(), 1);
        assert_eq!(portfolio.best(), (0, 3.0));
        assert!(portfolio.validate(ticket).is_err());
        let fresh = portfolio
            .issue(
                0,
                RoundId {
                    owner: 4,
                    proposal: 10,
                    incumbent: 3,
                },
            )
            .unwrap();
        assert_eq!(fresh.region.generation, 1);
    }
}
