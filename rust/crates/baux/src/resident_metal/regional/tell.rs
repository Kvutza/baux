use super::*;

pub(super) struct TellPlan {
    pub(super) source: usize,
    pub(super) routes: Vec<usize>,
    pub(super) staged: Option<Buffer>,
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
    if let Some(state) = states.get_mut(slot) {
        state.poisoned = true;
    }
    let retirement = ledger.retire(ticket);
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
    let routes = search
        .routes(Some(source), &candidate)
        .map_err(TellFailure::Retryable)?;
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
    Ok(TellPlan {
        source,
        routes,
        staged,
    })
}

pub(super) fn complete(
    search: &mut RegionalSearch,
    proposal: &RegionalProposal,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<NoisyDecision, TellFailure> {
    let TellPlan {
        source,
        routes,
        staged,
    } = prepare(search, proposal, observation)?;
    let state = search
        .states
        .get_mut(source)
        .ok_or_else(|| TellFailure::Retryable("unknown region".into()))?;
    let decision = match state.noisy_objectives(&proposal.proposal, observation) {
        Ok(decision) => decision,
        Err(error) if state.poisoned => {
            search
                .ledger
                .retire(proposal.ticket)
                .map_err(|error| TellFailure::Retired(error.to_string()))?;
            search.pending_proposals[source] = None;
            search.pending_tickets[source] = None;
            return Err(TellFailure::Retired(format!(
                "{error}; regional state was partially updated and its ticket was retired"
            )));
        }
        Err(error) => return Err(TellFailure::Retryable(error)),
    };
    if let Err(error) = state.sync() {
        state.poisoned = true;
        search
            .ledger
            .retire(proposal.ticket)
            .map_err(|error| TellFailure::Retired(error.to_string()))?;
        search.pending_proposals[source] = None;
        search.pending_tickets[source] = None;
        return Err(TellFailure::Retired(format!(
            "{error}; resident tell was dispatched and must not be retried"
        )));
    }
    let vector = observation.is_vector();
    let incumbent = if vector {
        search.ledger.best(source).unwrap_or(0.0) + decision.improvement.max(0.0)
    } else {
        f64::from(state.best().map_err(|error| {
            state.poisoned = true;
            TellFailure::Retired(format!(
                "{error}; resident tell completed but ticket must be retired"
            ))
        })?)
    };
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
    if let Some(stable) = staged {
        for slot in routes {
            search.pending_imports[slot].push(PendingImport {
                row: stable.clone(),
                observation,
                from_region: true,
            });
        }
    }
    Ok(decision)
}
