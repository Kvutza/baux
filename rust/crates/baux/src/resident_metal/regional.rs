//! Multiple resident trajectories using the shared transaction ledger.
//! Each engine retains its own incumbent/history/scratch; Metal runtime and
//! compiled pipelines are shared. This adapter does not claim cross-region
//! fused multi-region kernel.

use super::*;
use crate::{
    config::{
        ObservationSharing, RegionAllocation, RegionController, RegionMemory, RegionalSearchSpec,
        RestartCenter,
    },
    optimizer::multi_tr::SharingPolicy,
};
use baux::semantics::optimizer::portfolio::{Allocation, RegionTicket, RegionalLedger};

pub mod objectives;
mod tell;

struct PendingImport {
    row: Buffer,
    observation: crate::objective_observation::ObjectiveObservation,
}

#[derive(Clone, Debug)]
pub struct RegionalProposal {
    ticket: RegionTicket,
    proposal: Proposals,
}

/// Whether a failed tell can safely be retried with the same observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TellFailure {
    /// Validation failed before the resident engine mutated; the ticket remains
    /// pending.
    Retryable(String),
    /// Mutation may have started; the ticket was retired and must not be
    /// replayed.
    Retired(String),
    /// The ticket belongs to a completed, cancelled, or replaced generation.
    Stale(String),
}

#[cfg(test)]
mod tests;

impl RegionalProposal {
    pub fn ticket(&self) -> RegionTicket {
        self.ticket
    }
}

pub struct RegionalSearch {
    ledger: RegionalLedger,
    states: Vec<SearchState>,
    sharing: SharingPolicy,
    pending_imports: Vec<Vec<PendingImport>>,
    pending_proposals: Vec<Option<Proposals>>,
    pending_tickets: Vec<Option<RegionTicket>>,
    delivered_imports: u64,
}

impl RegionalSearch {
    /// Build resident regional search from the same typed policy as host
    /// search. Fields not owned by this adapter are rejected explicitly.
    pub fn from_spec(
        mut states: Vec<SearchState>,
        spec: &RegionalSearchSpec,
    ) -> Result<Self, String> {
        spec.validate()?;
        if states.len() != spec.count {
            return Err("regional search state count does not match its policy".into());
        }
        if spec.controller != RegionController::Turbo {
            return Err("resident regional search supports only the TuRBO controller".into());
        }
        if spec.restart_memory != RegionMemory::Fresh
            || spec.restart_candidates != 1
            || spec.restart_center != RestartCenter::First
        {
            return Err(
                "resident restart policy is caller-owned and cannot be configured here".into(),
            );
        }
        if !spec.acquisitions.is_empty() {
            objectives::configure(&mut states, &spec.acquisitions)?;
        }
        if spec.success_tolerance != 3 {
            return Err("resident regional search requires success-tolerance = 3".into());
        }
        let failures = usize::try_from(spec.failure_tolerance)
            .map_err(|_| "regional failure tolerance must fit usize")?;
        for state in &mut states {
            state.set_tolerance(failures)?;
        }
        let allocation = match spec.allocation {
            RegionAllocation::UtilityUcb => Allocation::Ucb { exploration: 1.0 },
            RegionAllocation::Balanced => Allocation::Balanced,
        };
        let sharing = match spec.sharing {
            ObservationSharing::Shared => SharingPolicy::Shared,
            ObservationSharing::NearestCenter => SharingPolicy::NearestCenter,
            ObservationSharing::Independent => SharingPolicy::Independent,
        };
        Self::with_sharing(states, allocation, sharing)
    }

    pub fn new(states: Vec<SearchState>, allocation: Allocation) -> Result<Self, String> {
        Self::with_sharing(states, allocation, SharingPolicy::Independent)
    }

    pub fn with_sharing(
        states: Vec<SearchState>,
        allocation: Allocation,
        sharing: SharingPolicy,
    ) -> Result<Self, String> {
        let dimensions = states
            .first()
            .ok_or("regional search needs a starting state")?
            .dimensions;
        if states
            .iter()
            .any(|state| state.dimensions != dimensions || state.axis.is_some())
        {
            return Err(
                "resident sharing requires equal weight dimensions and no scalar axis".into(),
            );
        }
        let starts = states
            .iter()
            .map(|state| {
                state.check_idle()?;
                let best = if state
                    .incumbent_objectives()?
                    .is_some_and(|record| record.observation.is_vector())
                {
                    0.0
                } else {
                    f64::from(state.best()?)
                };
                Ok(best)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let count = starts.len();
        Ok(Self {
            ledger: RegionalLedger::new(&starts, allocation).map_err(|error| error.to_string())?,
            states,
            sharing,
            pending_imports: (0..count).map(|_| Vec::new()).collect(),
            pending_proposals: (0..count).map(|_| None).collect(),
            pending_tickets: (0..count).map(|_| None).collect(),
            delivered_imports: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.ledger.len()
    }
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
    pub fn trials(&self) -> Vec<u64> {
        self.ledger.trials()
    }

    pub fn best(&self) -> (usize, f64) {
        self.ledger.best_region()
    }

    pub fn ask(&mut self, seed: u64, config: RoundConfig) -> Result<RegionalProposal, String> {
        let eligible = vec![true; self.len()];
        let slot = self
            .ledger
            .allocate(1, &eligible)
            .map_err(|error| error.to_string())?[0]
            .region;
        self.ask_region(slot, seed, config)
    }

    /// Explicit region choice permits other allocation policies without
    /// changing the execution or completion contract.
    pub fn ask_region(
        &mut self,
        slot: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<RegionalProposal, String> {
        self.flush_region(slot)?;
        let ticket = self.ledger.issue(slot).map_err(|error| error.to_string())?;
        let proposal = match self.states.get_mut(slot) {
            Some(state) => {
                let pool = state.pool_layout();
                state.ask_round(pool.arms() as usize, pool.slots() as usize, seed, config)
            }
            None => Err("unknown region".into()),
        };
        let proposal = match proposal {
            Ok(proposal) => proposal,
            Err(error) => {
                self.ledger
                    .retire(ticket)
                    .map_err(|retire| format!("{error}; ticket retirement failed: {retire}"))?;
                return Err(error);
            }
        };
        self.pending_proposals[slot] = Some(proposal.clone());
        self.pending_tickets[slot] = Some(ticket);
        Ok(RegionalProposal { ticket, proposal })
    }

    /// Borrowed resident storage. Consumers must finish before tell/restart.
    pub fn buffer(&self, proposal: &RegionalProposal) -> Result<Buffer, String> {
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.states
            .get(proposal.ticket.region.slot)
            .ok_or("unknown region")?
            .propose_buffer(&proposal.proposal)
    }

    pub fn incumbent(&self, slot: usize) -> Result<Buffer, String> {
        let state = self.states.get(slot).ok_or("unknown region")?;
        state.check_healthy()?;
        Ok(state.base_buffer())
    }

    pub fn dimensions(&self, slot: usize) -> Result<usize, String> {
        Ok(self.states.get(slot).ok_or("unknown region")?.len())
    }

    pub fn tell(
        &mut self,
        proposal: &RegionalProposal,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        self.tell_result(proposal, value, variance)
            .map_err(|failure| match failure {
                TellFailure::Retryable(message)
                | TellFailure::Retired(message)
                | TellFailure::Stale(message) => message,
            })
    }

    /// Tell with an explicit retry contract. Pre-mutation failures preserve the
    /// pending ticket; failures after mutation retire it to prevent double
    /// count.
    pub fn tell_result(
        &mut self,
        proposal: &RegionalProposal,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, TellFailure> {
        tell::complete(
            self,
            proposal,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Retire an unevaluated proposal without updating regional statistics.
    pub fn cancel(&mut self, proposal: &RegionalProposal) -> Result<(), String> {
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.states
            .get_mut(proposal.ticket.region.slot)
            .ok_or("unknown region")?
            .cancel_round(&proposal.proposal)?;
        self.ledger
            .retire(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.pending_proposals[proposal.ticket.region.slot] = None;
        self.pending_tickets[proposal.ticket.region.slot] = None;
        Ok(())
    }

    /// Admit an externally evaluated resident row to one target model. This
    /// carries no proposal origin and grants no controller or trial credit.
    /// A busy target receives a stable GPU copy after its pending tell.
    pub fn import_region(
        &mut self,
        slot: usize,
        row: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<(), String> {
        objectives::import(
            self,
            slot,
            row,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    /// Route one external row through the configured relevance policy. Rows
    /// admitted to recipients are staged in resident storage and delivered
    /// before each recipient's next ask or on explicit flush.
    pub fn import_external(
        &mut self,
        row: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<Vec<usize>, String> {
        check_scores(&[value], &[variance])?;
        if self.sharing == SharingPolicy::Independent {
            return Err("independent sharing needs an explicit import_region target".into());
        }
        let routes = self.routes(None, row)?;
        for &slot in &routes {
            if self.pending_imports[slot].len() == MAX_HISTORY {
                return Err("resident sharing queue is full; flush a recipient".into());
            }
        }
        if routes.is_empty() {
            return Ok(routes);
        }
        let state = self.states.get(routes[0]).ok_or("unknown region")?;
        let stable = state.runtime.buffer::<u16>(state.dimensions);
        state.copy(row, &stable)?;
        for &slot in &routes {
            self.pending_imports[slot].push(PendingImport {
                row: stable.clone(),
                observation: crate::objective_observation::ObjectiveObservation::scalar_adapter(
                    value, variance,
                ),
            });
        }
        Ok(routes)
    }

    fn routes(&self, source: Option<usize>, incoming: &Buffer) -> Result<Vec<usize>, String> {
        if self.sharing == SharingPolicy::Independent {
            return Ok(Vec::new());
        }
        let mut centers = Vec::with_capacity(self.len());
        for slot in 0..self.len() {
            let state = self.states.get(slot).ok_or("unknown region")?;
            centers.push(state.measure_import(incoming)?.center);
        }
        match self.sharing {
            SharingPolicy::Independent => Ok(Vec::new()),
            SharingPolicy::NearestCenter => {
                let nearest = centers
                    .iter()
                    .enumerate()
                    .min_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| a.cmp(b)))
                    .map(|(slot, _)| slot)
                    .ok_or("regional search has no center")?;
                Ok((Some(nearest) != source)
                    .then_some(nearest)
                    .into_iter()
                    .collect())
            }
            SharingPolicy::Shared => {
                let mut routes = Vec::new();
                for (slot, &distance) in centers.iter().enumerate() {
                    if Some(slot) == source {
                        continue;
                    }
                    let state = self.states.get(slot).ok_or("unknown region")?;
                    // BAUX resident policy: the trust length defines a radius
                    // in the same block-weighted squared metric as acquisition.
                    let scale = state
                        .blocks
                        .iter()
                        .map(|block| {
                            block.len as f64
                                * f64::from(block.weight)
                                * f64::from(block.scale).powi(2)
                        })
                        .sum::<f64>();
                    let radius2 = state.length.powi(2) * scale;
                    if f64::from(distance) <= radius2 {
                        routes.push(slot);
                    }
                }
                Ok(routes)
            }
        }
    }

    fn flush_region(&mut self, slot: usize) -> Result<(), String> {
        let state = self.states.get_mut(slot).ok_or("unknown region")?;
        state.check_idle()?;
        while let Some(entry) = self.pending_imports[slot].first() {
            state.import_objectives(&entry.row, entry.observation)?;
            self.pending_imports[slot].remove(0);
            self.delivered_imports += 1;
        }
        Ok(())
    }

    pub fn sharing_backlog(&self) -> Vec<usize> {
        self.pending_imports.iter().map(Vec::len).collect()
    }

    pub fn delivered_imports(&self) -> u64 {
        self.delivered_imports
    }

    pub fn flush_sharing(&mut self, slot: usize) -> Result<(), String> {
        self.flush_region(slot)
    }

    /// Replace one trajectory. If it has an outstanding proposal, cancel that
    /// engine round and invalidate its ticket; late results are stale. Queued
    /// evidence admitted to the old generation is discarded.
    pub fn restart(&mut self, slot: usize, state: SearchState) -> Result<(), String> {
        state.check_idle()?;
        let best = f64::from(state.best()?);
        let pending = self
            .pending_proposals
            .get(slot)
            .ok_or("unknown region")?
            .clone();
        if let Some(old_proposal) = pending {
            let ticket = self
                .pending_tickets
                .get(slot)
                .copied()
                .flatten()
                .ok_or("region has no outstanding ticket")?;
            self.ledger
                .validate(ticket)
                .map_err(|error| error.to_string())?;
            let old = self.states.get_mut(slot).ok_or("unknown region")?;
            if !old.poisoned {
                old.cancel_round(&old_proposal)?;
            }
            self.ledger
                .restart(slot, best)
                .map_err(|error| error.to_string())?;
            self.pending_proposals[slot] = None;
        } else {
            self.ledger
                .restart(slot, best)
                .map_err(|error| error.to_string())?;
        }
        self.states[slot] = state;
        self.pending_imports[slot].clear();
        self.pending_tickets[slot] = None;
        Ok(())
    }

    pub fn memory_bytes(&self) -> u64 {
        let engines: u64 = self
            .states
            .iter()
            .map(|state| state.memory_info().resident_bytes)
            .sum();
        let queued = self
            .pending_imports
            .iter()
            .flatten()
            .map(|entry| entry.row.length())
            .sum::<u64>();
        engines + queued
    }
}
