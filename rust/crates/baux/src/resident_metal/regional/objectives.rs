//! Vector-observation adapter for resident regional search.

use super::*;

pub(super) fn configure(
    states: Vec<SearchState>,
    acquisitions: &[Option<crate::config::AcquisitionSpec>],
) -> Result<Vec<SearchState>, String> {
    states
        .into_iter()
        .zip(acquisitions)
        .map(|(state, acquisition)| match acquisition {
            Some(crate::config::AcquisitionSpec::Pareto { scales }) => {
                state.configure_objectives(Some(crate::config::ObjectivePolicy {
                    acquisition: crate::config::ObjectiveAcquisition::Pareto,
                    scales: scales.clone(),
                }))
            }
            None => Ok(state),
            Some(_) => {
                Err("resident scalar regional acquisitions belong to each ask configuration".into())
            }
        })
        .collect()
}

/// Complete a proposal with independently measured objectives. Pareto
/// improvement drives allocation while the explicit control estimate drives
/// trust-region adaptation.
pub fn tell(
    search: RegionalSearch,
    proposal: &RegionalProposal,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<(RegionalSearch, NoisyDecision), String> {
    super::tell::complete(search, proposal, observation).map_err(|failure| match failure {
        TellFailure::Retryable(message)
        | TellFailure::Retired(message)
        | TellFailure::Stale(message) => message,
    })
}

/// Import vector evidence without granting proposal or controller credit.
pub fn import(
    mut search: RegionalSearch,
    slot: usize,
    row: &Buffer,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<RegionalSearch, String> {
    observation.validate()?;
    let (is_idle, row_bytes, dimensions, runtime) = {
        let state = search.states.get(slot).ok_or("unknown region")?;
        state.check_healthy()?;
        if !state.has_surrogate() {
            return Err("surrogate-free region has no local model history to import into".into());
        }
        (
            state.check_idle().is_ok(),
            state.row_bytes(),
            state.dimensions,
            Arc::clone(&state.runtime),
        )
    };
    if row.length() != row_bytes {
        return Err("Imported resident row has the wrong byte length".into());
    }
    if is_idle {
        let state_ref = search.states.get_mut(slot).ok_or("unknown region")?;
        let state = std::mem::replace(state_ref, state_ref.clone());
        *state_ref = state.import_objectives(row, observation)?;
        search.delivered_imports += 1;
        return Ok(search);
    }
    if search.pending_imports[slot].len() == MAX_HISTORY {
        return Err("resident sharing queue is full; complete the pending recipient".into());
    }
    let stable = runtime.buffer::<u16>(dimensions);
    search.states[slot].copy(row, &stable)?;
    search.pending_imports[slot].push(PendingImport {
        row: stable,
        observation,
        from_region: false,
    });
    Ok(search)
}
