use super::*;
use std::sync::Arc;

pub(super) struct TellPlan {
    pub(super) source: usize,
    pub(super) routes: Vec<usize>,
    pub(super) staged: Option<Buffer>,
    pub(super) archive_copy: Option<Arc<Buffer>>,
}

pub(super) fn retire_aftertell(
    ledger: &mut RegionalLedger,
    states: &mut [SearchState],
    pending_proposals: &mut [Option<Proposals>],
    pending_tickets: &mut [Option<RegionTicket>],
    slot: usize,
    ticket: RegionTicket,
    error: String,
) -> TellFailure {
    if let Some(state_ref) = states.get_mut(slot)
        && let Some(prop) = pending_proposals.get(slot).and_then(Option::as_ref)
    {
        let clean = state_ref.clone();
        if let Ok(cancelled) = clean.cancel_round(prop) {
            *state_ref = cancelled;
        }
    }
    let retirement = ledger.retire(ticket);
    pending_proposals[slot] = None;
    pending_tickets[slot] = None;
    let detail = match retirement {
        Ok(()) => format!("{error}; ticket retired without poisoning resident engine"),
        Err(retire_error) => {
            format!("{error}; ticket retirement also failed: {retire_error}")
        }
    };
    TellFailure::Retired(detail)
}

fn prepare(
    search: &RegionalSearch,
    proposal: &RegionalProposal,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<TellPlan, TellFailure> {
    let source = proposal.ticket.region.slot;
    if search.pending_tickets.get(source).copied().flatten() != Some(proposal.ticket) {
        return Err(TellFailure::Stale(
            "regional proposal is no longer pending in its generation".into(),
        ));
    }
    search
        .ledger
        .validate(proposal.ticket)
        .map_err(|error| TellFailure::Stale(error.to_string()))?;
    observation.validate().map_err(TellFailure::Retryable)?;
    let state = search
        .states
        .get(source)
        .ok_or_else(|| TellFailure::Retryable("unknown region".into()))?;
    let candidate = state
        .propose_buffer(&proposal.proposal)
        .map_err(TellFailure::Retryable)?;
    let routes =
        routing::routes(search, Some(source), &candidate).map_err(TellFailure::Retryable)?;
    if routes
        .iter()
        .any(|&slot| search.pending_imports[slot].len() == MAX_HISTORY)
    {
        return Err(TellFailure::Retryable(
            "resident sharing queue is full; complete the pending recipient".into(),
        ));
    }
    let staged = if routes.is_empty() {
        None
    } else {
        let stable = state.runtime.buffer::<u16>(state.dimensions);
        state
            .copy(&candidate, &stable)
            .map_err(TellFailure::Retryable)?;
        Some(stable)
    };
    let archive_copy = if search.restart_memory == RegionMemory::RetainRelevant {
        let snap = state.runtime.buffer::<u16>(state.dimensions);
        state
            .copy(&candidate, &snap)
            .map_err(TellFailure::Retryable)?;
        Some(Arc::new(snap))
    } else {
        None
    };
    Ok(TellPlan {
        source,
        routes,
        staged,
        archive_copy,
    })
}

pub(super) fn complete(
    mut search: RegionalSearch,
    proposal: &RegionalProposal,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<(RegionalSearch, NoisyDecision), TellFailure> {
    let TellPlan {
        source,
        routes,
        staged,
        archive_copy,
    } = prepare(&search, proposal, observation)?;
    if source >= search.states.len() {
        return Err(TellFailure::Retryable("unknown region".into()));
    }
    let state = search.states.remove(source);
    let backup = state.clone();
    let outcome = if state.implicit_history && state.history < state.initial_observations {
        state.initial_objectives(&proposal.proposal, observation)
    } else if state.implicit_history {
        state.modeled_objectives(&proposal.proposal, observation)
    } else {
        state.noisy_objectives(&proposal.proposal, observation)
    };
    let (state, decision) = match outcome {
        Ok((state, decision)) => (state, decision),
        Err(error) => {
            search.states.insert(source, backup);
            return Err(TellFailure::Retryable(error));
        }
    };
    let pre_sync = backup;
    let state = match state.sync() {
        Ok((synced, _queued)) => synced,
        Err(error) => {
            let restored = pre_sync
                .clone()
                .cancel_round(&proposal.proposal)
                .unwrap_or(pre_sync);
            search.states.insert(source, restored);
            search
                .ledger
                .retire(proposal.ticket)
                .map_err(|error| TellFailure::Retired(error.to_string()))?;
            search.pending_proposals[source] = None;
            search.pending_tickets[source] = None;
            return Err(TellFailure::Retired(format!(
                "{error}; resident tell failed sync and its ticket was retired"
            )));
        }
    };
    let vector = observation.is_vector();
    let incumbent = if vector {
        search.ledger.best(source).unwrap_or(0.0) + decision.improvement.max(0.0)
    } else {
        match state.best() {
            Ok(best) => f64::from(best),
            Err(error) => {
                search.states.insert(source, state);
                return Err(TellFailure::Retired(format!(
                    "{error}; resident tell completed but ticket must be retired"
                )));
            }
        }
    };
    search.states.insert(source, state);
    let utility = if vector {
        decision.improvement
    } else {
        f64::from(observation.control().mean)
    };
    if let Err(error) = search
        .ledger
        .complete(proposal.ticket, &[utility], incumbent)
    {
        return Err(retire_aftertell(
            &mut search.ledger,
            &mut search.states,
            &mut search.pending_proposals,
            &mut search.pending_tickets,
            source,
            proposal.ticket,
            error.to_string(),
        ));
    }
    search.pending_proposals[source] = None;
    search.pending_tickets[source] = None;
    if let Some(arc_row) = archive_copy {
        if search.retained_archive.len() >= 2 * MAX_HISTORY {
            search.retained_archive.remove(0);
        }
        search.retained_archive.push((arc_row, observation));
    }
    if let Some(stable) = staged {
        for slot in routes {
            search.pending_imports[slot].push(PendingImport {
                row: stable.clone(),
                observation,
                from_region: true,
            });
        }
    }
    Ok((search, decision))
}
