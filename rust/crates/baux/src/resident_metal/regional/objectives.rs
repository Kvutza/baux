//! Vector-observation adapter for resident regional search.

use super::*;

pub(super) fn configure(
    states: &mut [SearchState],
    acquisitions: &[Option<crate::config::AcquisitionSpec>],
) -> Result<(), String> {
    for (state, acquisition) in states.iter_mut().zip(acquisitions) {
        match acquisition {
            Some(crate::config::AcquisitionSpec::Pareto { scales }) => {
                state.configure_objectives(Some(crate::config::ObjectivePolicy {
                    acquisition: crate::config::ObjectiveAcquisition::Pareto,
                    scales: scales.clone(),
                }))?;
            }
            None => {}
            Some(_) => {
                return Err(
                    "resident scalar regional acquisitions belong to each ask configuration".into(),
                );
            }
        }
    }
    Ok(())
}

/// Complete a proposal with independently measured objectives. Pareto
/// improvement drives allocation while the explicit control estimate drives
/// trust-region adaptation.
pub fn tell(
    search: &mut RegionalSearch,
    proposal: &RegionalProposal,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<NoisyDecision, String> {
    super::tell::complete(search, proposal, observation).map_err(|failure| match failure {
        TellFailure::Retryable(message)
        | TellFailure::Retired(message)
        | TellFailure::Stale(message) => message,
    })
}

/// Import vector evidence without granting proposal or controller credit.
pub fn import(
    search: &mut RegionalSearch,
    slot: usize,
    row: &Buffer,
    observation: crate::objective_observation::ObjectiveObservation,
) -> Result<(), String> {
    observation.validate()?;
    let state = search.states.get(slot).ok_or("unknown region")?;
    state.check_healthy()?;
    if !state.has_surrogate() {
        return Err("surrogate-free region has no local model history to import into".into());
    }
    if row.length() != state.row_bytes() {
        return Err("Imported resident row has the wrong byte length".into());
    }
    if state.check_idle().is_ok() {
        search
            .states
            .get_mut(slot)
            .ok_or("unknown region")?
            .import_objectives(row, observation)?;
        search.delivered_imports += 1;
        return Ok(());
    }
    if search.pending_imports[slot].len() == MAX_HISTORY {
        return Err("resident sharing queue is full; complete the pending recipient".into());
    }
    let stable = state.runtime.buffer::<u16>(state.dimensions);
    state.copy(row, &stable)?;
    search.pending_imports[slot].push(PendingImport {
        row: stable,
        observation,
        from_region: false,
    });
    Ok(())
}
