//! Multiple resident trajectories using the shared transaction ledger.
//! Each engine retains its own incumbent/history/scratch; Metal runtime and
//! compiled pipelines are shared. This adapter does not claim cross-region
//! fused multi-region kernel.

use super::*;
use crate::optimizer::multi_tr::SharingPolicy;
use baux::semantics::optimizer::{
    RoundId,
    portfolio::{Allocation, Portfolio, RegionTicket},
};

struct PendingImport {
    row: Buffer,
    value: f32,
    variance: f32,
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

fn retire_aftertell(
    regions: &mut Portfolio<SearchState>,
    pending_proposals: &mut [Option<Proposals>],
    pending_tickets: &mut [Option<RegionTicket>],
    slot: usize,
    ticket: RegionTicket,
    error: String,
) -> TellFailure {
    if let Ok(state) = regions.engine_mut(slot) {
        state.poisoned = true;
    }
    let retirement = regions.retire(ticket);
    pending_proposals[slot] = None;
    pending_tickets[slot] = None;
    let detail = match retirement {
        Ok(()) => format!("{error}; resident engine poisoned and ticket retired"),
        Err(retire_error) => format!(
            "{error}; resident engine poisoned, ticket retirement also failed: {retire_error}"
        ),
    };
    TellFailure::Retired(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_cleanup() {
        let state = SearchState::new_fp16(
            &[0x3400; 4],
            vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()],
            2,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            ProposalKernel::Gaussian,
        )
        .unwrap();
        let mut regions = Portfolio::new(vec![(state, -1.0)], Allocation::Balanced).unwrap();
        let ticket = regions
            .issue(
                0,
                RoundId {
                    owner: 1,
                    proposal: 2,
                    incumbent: 0,
                },
            )
            .unwrap();
        let mut proposals = vec![None];
        let mut tickets = vec![Some(ticket)];

        let failure = retire_aftertell(
            &mut regions,
            &mut proposals,
            &mut tickets,
            0,
            ticket,
            "injected ledger completion failure".into(),
        );

        assert!(matches!(failure, TellFailure::Retired(_)));
        assert!(regions.engine(0).unwrap().poisoned);
        assert!(regions.validate(ticket).is_err());
        assert!(proposals[0].is_none());
        assert!(tickets[0].is_none());
    }
}

impl RegionalProposal {
    pub fn ticket(&self) -> RegionTicket {
        self.ticket
    }
}

pub struct RegionalSearch {
    regions: Portfolio<SearchState>,
    sharing: SharingPolicy,
    pending_imports: Vec<Vec<PendingImport>>,
    pending_proposals: Vec<Option<Proposals>>,
    pending_tickets: Vec<Option<RegionTicket>>,
    delivered_imports: u64,
}

impl RegionalSearch {
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
            .into_iter()
            .map(|state| {
                state.check_idle()?;
                let best = f64::from(state.best()?);
                Ok((state, best))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let count = starts.len();
        Ok(Self {
            regions: Portfolio::new(starts, allocation)?,
            sharing,
            pending_imports: (0..count).map(|_| Vec::new()).collect(),
            pending_proposals: (0..count).map(|_| None).collect(),
            pending_tickets: (0..count).map(|_| None).collect(),
            delivered_imports: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.regions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
    pub fn trials(&self) -> Vec<u64> {
        self.regions.trials()
    }

    pub fn best(&self) -> (usize, f64) {
        self.regions.best()
    }

    pub fn ask(&mut self, seed: u64, config: RoundConfig) -> Result<RegionalProposal, String> {
        let slot = self.regions.choose()?;
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
        let state = self.regions.engine_mut(slot)?;
        let pool = state.pool_layout();
        let proposal =
            state.ask_round(pool.arms() as usize, pool.slots() as usize, seed, config)?;
        let ticket = self.regions.issue(
            slot,
            RoundId {
                owner: proposal.owner,
                proposal: proposal.id,
                incumbent: proposal.base_id,
            },
        )?;
        self.pending_proposals[slot] = Some(proposal.clone());
        self.pending_tickets[slot] = Some(ticket);
        Ok(RegionalProposal { ticket, proposal })
    }

    /// Borrowed resident storage. Consumers must finish before tell/restart.
    pub fn buffer(&self, proposal: &RegionalProposal) -> Result<Buffer, String> {
        self.regions.validate(proposal.ticket)?;
        self.regions
            .engine(proposal.ticket.region.slot)?
            .propose_buffer(&proposal.proposal)
    }

    pub fn incumbent(&self, slot: usize) -> Result<Buffer, String> {
        let state = self.regions.engine(slot)?;
        state.check_healthy()?;
        Ok(state.base_buffer())
    }

    pub fn dimensions(&self, slot: usize) -> Result<usize, String> {
        Ok(self.regions.engine(slot)?.len())
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
        if self
            .pending_tickets
            .get(proposal.ticket.region.slot)
            .copied()
            .flatten()
            != Some(proposal.ticket)
        {
            return Err(TellFailure::Stale(
                "regional proposal is no longer pending in its generation".into(),
            ));
        }
        self.regions
            .validate(proposal.ticket)
            .map_err(TellFailure::Stale)?;
        check_scores(&[value], &[variance]).map_err(TellFailure::Retryable)?;
        let source = proposal.ticket.region.slot;
        let candidate = self
            .regions
            .engine(source)
            .map_err(TellFailure::Retryable)?
            .propose_buffer(&proposal.proposal)
            .map_err(TellFailure::Retryable)?;
        let routes = self
            .routes(Some(source), &candidate)
            .map_err(TellFailure::Retryable)?;
        for &slot in &routes {
            if self.pending_imports[slot].len() == MAX_HISTORY {
                return Err(TellFailure::Retryable(
                    "resident sharing queue is full; complete the pending recipient".into(),
                ));
            }
        }
        // The source proposal scratch may be reused on its next ask. One GPU
        // copy keeps a stable row for every recipient without host readback.
        let staged = if routes.is_empty() {
            None
        } else {
            let state = self
                .regions
                .engine(source)
                .map_err(TellFailure::Retryable)?;
            let stable = state.runtime.buffer::<u16>(state.dimensions);
            state
                .copy(&candidate, &stable)
                .map_err(TellFailure::Retryable)?;
            Some(stable)
        };
        let state = self
            .regions
            .engine_mut(source)
            .map_err(TellFailure::Retryable)?;
        let decision = match state.tell_noisy(&proposal.proposal, value, variance) {
            Ok(decision) => decision,
            Err(error) => {
                if state.poisoned {
                    self.regions
                        .retire(proposal.ticket)
                        .map_err(TellFailure::Retired)?;
                    self.pending_proposals[source] = None;
                    self.pending_tickets[source] = None;
                    return Err(TellFailure::Retired(format!(
                        "{error}; regional state was partially updated and its ticket was retired"
                    )));
                }
                return Err(TellFailure::Retryable(error));
            }
        };
        if let Err(error) = state.sync() {
            state.poisoned = true;
            self.regions
                .retire(proposal.ticket)
                .map_err(TellFailure::Retired)?;
            self.pending_proposals[source] = None;
            self.pending_tickets[source] = None;
            return Err(TellFailure::Retired(format!(
                "{error}; resident tell was dispatched and must not be retried"
            )));
        }
        let incumbent = match state.best() {
            Ok(incumbent) => f64::from(incumbent),
            Err(error) => {
                state.poisoned = true;
                self.regions
                    .retire(proposal.ticket)
                    .map_err(TellFailure::Retired)?;
                self.pending_proposals[source] = None;
                self.pending_tickets[source] = None;
                return Err(TellFailure::Retired(format!(
                    "{error}; resident tell completed but ticket was retired"
                )));
            }
        };
        if let Err(error) = self
            .regions
            .complete(proposal.ticket, f64::from(value), incumbent)
        {
            // The engine has already accepted this result, so a ledger failure
            // cannot be retried safely. Poison it, retire whatever ledger entry
            // remains, and clear adapter handles even if retirement also fails.
            return Err(retire_aftertell(
                &mut self.regions,
                &mut self.pending_proposals,
                &mut self.pending_tickets,
                source,
                proposal.ticket,
                error,
            ));
        }
        self.pending_proposals[source] = None;
        self.pending_tickets[source] = None;
        if let Some(stable) = staged {
            for slot in routes {
                self.pending_imports[slot].push(PendingImport {
                    row: stable.clone(),
                    value,
                    variance,
                });
            }
        }
        Ok(decision)
    }

    /// Retire an unevaluated proposal without updating regional statistics.
    pub fn cancel(&mut self, proposal: &RegionalProposal) -> Result<(), String> {
        self.regions.validate(proposal.ticket)?;
        self.regions
            .engine_mut(proposal.ticket.region.slot)?
            .cancel_round(&proposal.proposal)?;
        self.regions.retire(proposal.ticket)?;
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
        check_scores(&[value], &[variance])?;
        let state = self.regions.engine(slot)?;
        state.check_healthy()?;
        if row.length() != state.row_bytes() {
            return Err("Imported resident row has the wrong byte length".into());
        }
        if state.check_idle().is_ok() {
            self.regions
                .engine_mut(slot)?
                .import_observation(row, value, variance)?;
            self.delivered_imports += 1;
            return Ok(());
        }
        if self.pending_imports[slot].len() == MAX_HISTORY {
            return Err("resident sharing queue is full; complete the pending recipient".into());
        }
        let stable = state.runtime.buffer::<u16>(state.dimensions);
        state.copy(row, &stable)?;
        self.pending_imports[slot].push(PendingImport {
            row: stable,
            value,
            variance,
        });
        Ok(())
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
        let state = self.regions.engine(routes[0])?;
        let stable = state.runtime.buffer::<u16>(state.dimensions);
        state.copy(row, &stable)?;
        for &slot in &routes {
            self.pending_imports[slot].push(PendingImport {
                row: stable.clone(),
                value,
                variance,
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
            let state = self.regions.engine(slot)?;
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
                    let state = self.regions.engine(slot)?;
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
        let state = self.regions.engine_mut(slot)?;
        state.check_idle()?;
        while let Some(entry) = self.pending_imports[slot].first() {
            state.import_observation(&entry.row, entry.value, entry.variance)?;
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
            let old = self.regions.engine_mut(slot)?;
            if !old.poisoned {
                old.cancel_round(&old_proposal)?;
            }
            self.regions.restart_abandoning(ticket, state, best)?;
            self.pending_proposals[slot] = None;
        } else {
            self.regions.restart(slot, state, best)?;
        }
        self.pending_imports[slot].clear();
        self.pending_tickets[slot] = None;
        Ok(())
    }

    pub fn memory_bytes(&self) -> u64 {
        let engines: u64 = (0..self.len())
            .map(|slot| {
                self.regions
                    .engine(slot)
                    .unwrap()
                    .memory_info()
                    .resident_bytes
            })
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
