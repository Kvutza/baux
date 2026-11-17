//! Multiple resident trajectories using the shared transaction ledger.

use super::*;
use crate::{
    config::{
        ObservationSharing, RegionAllocation, RegionController, RegionMemory, RegionalSearchSpec,
        RestartCenter,
    },
    optimizer::multi_tr::SharingPolicy,
};
use baux::semantics::optimizer::portfolio::{Allocation, RegionTicket, RegionalLedger};
use std::sync::Arc;

pub mod objectives;
pub mod routing;
mod tell;

#[derive(Clone)]
struct PendingImport {
    row: Buffer,
    observation: crate::objective_observation::ObjectiveObservation,
    from_region: bool,
}

#[derive(Clone, Debug)]
pub struct RegionalProposal {
    ticket: RegionTicket,
    proposal: Proposals,
}

/// Whether a failed tell can safely be retried with the same observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TellFailure {
    Retryable(String),
    Retired(String),
    Stale(String),
}

#[cfg(test)]
mod tests;

impl RegionalProposal {
    pub fn ticket(&self) -> RegionTicket {
        self.ticket
    }

    pub fn proposal(&self) -> &Proposals {
        &self.proposal
    }
}

#[derive(Clone)]
pub struct RegionalSearch {
    ledger: RegionalLedger,
    states: Vec<SearchState>,
    sharing: SharingPolicy,
    joint_acquisition: bool,
    restart_memory: RegionMemory,
    restart_candidates: usize,
    restart_center: RestartCenter,
    region_acquisitions: Vec<Option<crate::config::AcquisitionSpec>>,
    retained_archive: Vec<(
        Arc<Buffer>,
        crate::objective_observation::ObjectiveObservation,
    )>,
    pending_imports: Vec<Vec<PendingImport>>,
    pending_proposals: Vec<Option<Proposals>>,
    pending_tickets: Vec<Option<RegionTicket>>,
    delivered_imports: u64,
}

fn encode_unit(u: f32, fp16: bool) -> u16 {
    if fp16 {
        let bits = u.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x03ff;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((exp as u16) << 10) | (mant as u16)
        }
    } else {
        let bits = u.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
}

impl RegionalSearch {
    /// Build resident regional search from the same typed policy as host
    /// search.
    pub fn from_spec(
        mut states: Vec<SearchState>,
        spec: &RegionalSearchSpec,
    ) -> Result<Self, String> {
        spec.validate()?;
        if states.len() != spec.count {
            return Err("regional search state count does not match its policy".into());
        }
        let fixed_radius = spec.controller == RegionController::FixedRadius;
        if !spec.acquisitions.is_empty() {
            states = objectives::configure(states, &spec.acquisitions)?;
        }
        let successes = usize::try_from(spec.success_tolerance)
            .ok()
            .filter(|&v| v > 0)
            .ok_or("regional success tolerance must be positive")?;
        let explicit_failures = usize::try_from(spec.failure_tolerance).ok();
        states = states
            .into_iter()
            .map(|state| {
                let failures = match explicit_failures {
                    Some(f) if f > 0 => f,
                    _ => SearchState::failure_tolerance(
                        state.dimensions,
                        state.pool_layout.arms() as usize,
                    ),
                };
                state
                    .set_tolerance(failures)?
                    .set_successes(successes)?
                    .fix_radius(fixed_radius)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let allocation = match spec.allocation {
            RegionAllocation::IncumbentUcb1 => Allocation::IncumbentUcb1 { exploration: 1.0 },
            RegionAllocation::LeastTrials => Allocation::LeastTrials,
            RegionAllocation::JointAcquisition => Allocation::JointAcquisition,
        };
        let sharing = match spec.sharing {
            ObservationSharing::BoxContain => SharingPolicy::BoxContain,
            ObservationSharing::EllipsoidBall => SharingPolicy::EllipsoidBall,
            ObservationSharing::NearestCenter => SharingPolicy::NearestCenter,
            ObservationSharing::OriginOnly => SharingPolicy::OriginOnly,
        };
        let mut search = Self::with_sharing(states, allocation, sharing)?;
        search.restart_memory = spec.restart_memory;
        search.restart_candidates = spec.restart_candidates.max(1);
        search.restart_center = spec.restart_center;
        search.region_acquisitions = spec.acquisitions.clone();
        if search.restart_memory == RegionMemory::RetainRelevant {
            for st in &search.states {
                if st.history > 0 {
                    let snap = st.runtime.buffer::<u16>(st.dimensions);
                    st.copy(&st.base, &snap)?;
                    let obs = st
                        .incumbent_objectives()?
                        .map(|r| r.observation)
                        .unwrap_or_else(|| {
                            crate::objective_observation::ObjectiveObservation::scalar_adapter(
                                st.best,
                                st.best_variance,
                            )
                        });
                    search.retained_archive.push((Arc::new(snap), obs));
                }
            }
        }
        Ok(search)
    }

    pub fn new(states: Vec<SearchState>, allocation: Allocation) -> Result<Self, String> {
        Self::with_sharing(states, allocation, SharingPolicy::OriginOnly)
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
        let joint_acquisition = matches!(allocation, Allocation::JointAcquisition);
        Ok(Self {
            ledger: RegionalLedger::new(&starts, allocation).map_err(|error| error.to_string())?,
            states,
            sharing,
            joint_acquisition,
            restart_memory: RegionMemory::Fresh,
            restart_candidates: 1,
            restart_center: RestartCenter::First,
            region_acquisitions: Vec::new(),
            retained_archive: Vec::new(),
            pending_imports: (0..count).map(|_| Vec::new()).collect(),
            pending_proposals: (0..count).map(|_| None).collect(),
            pending_tickets: (0..count).map(|_| None).collect(),
            delivered_imports: 0,
        })
    }

    pub fn fork(&self) -> Result<Self, String> {
        Ok(self.clone())
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

    pub fn ask(
        mut self,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, RegionalProposal), String> {
        if self.joint_acquisition {
            let mut candidates: Vec<(usize, SearchState, Proposals)> = Vec::new();
            let mut eligible = vec![false; self.len()];
            for slot in 0..self.len() {
                if self.pending_tickets[slot].is_some() || self.states[slot].check_idle().is_err() {
                    continue;
                }
                let r_cfg = objectives::region_acquisition(
                    config,
                    self.region_acquisitions.get(slot).and_then(Option::as_ref),
                );
                let adopt = r_cfg.acquisition.kind == AcquisitionKind::Random;
                self = routing::flush_region(self, slot, adopt)?;
                let pool = self.states[slot].pool_layout();
                let r_seed = crate::hash::splitmix64(
                    seed ^ ((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                );
                let branched = self.states[slot].clone();
                if let Ok((asked_state, prop)) =
                    branched.ask_round(pool.arms() as usize, pool.slots() as usize, r_seed, r_cfg)
                {
                    eligible[slot] = true;
                    candidates.push((slot, asked_state, prop));
                }
            }
            if candidates.is_empty() {
                return Err(
                    "no eligible idle region produced a finite joint-acquisition candidate".into(),
                );
            }
            let scores: Vec<f64> = candidates
                .iter()
                .map(|(_, _, p)| f64::from(p.score))
                .collect();
            let regions: Vec<usize> = candidates.iter().map(|(s, _, _)| *s).collect();
            let mut chosen = [0usize; 1];
            let n_sel = baux::semantics::optimizer::portfolio::joint_select(
                &scores,
                &regions,
                &eligible,
                baux::semantics::key::Key::new(seed),
                &mut chosen,
            );
            let win_pos = if n_sel > 0 { chosen[0] } else { 0 };
            let (win_slot, win_state, win_prop) = candidates.swap_remove(win_pos);
            let ticket = self
                .ledger
                .issue(win_slot)
                .map_err(|error| error.to_string())?;
            self.states[win_slot] = win_state;
            self.pending_proposals[win_slot] = Some(win_prop.clone());
            self.pending_tickets[win_slot] = Some(ticket);
            return Ok((
                self,
                RegionalProposal {
                    ticket,
                    proposal: win_prop,
                },
            ));
        }
        let eligible: Vec<bool> = (0..self.len())
            .map(|slot| {
                self.pending_tickets[slot].is_none() && self.states[slot].check_idle().is_ok()
            })
            .collect();
        let slot = self
            .ledger
            .allocate(1, &eligible)
            .map_err(|error| error.to_string())?[0]
            .region;
        self.ask_region(slot, seed, config)
    }

    pub fn ask_region(
        self,
        slot: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, RegionalProposal), String> {
        let config = objectives::region_acquisition(
            config,
            self.region_acquisitions.get(slot).and_then(Option::as_ref),
        );
        let adopt = config.acquisition.kind == AcquisitionKind::Random;
        let mut search = routing::flush_region(self, slot, adopt)?;
        let ticket = search
            .ledger
            .issue(slot)
            .map_err(|error| error.to_string())?;
        let state_ref = match search.states.get_mut(slot) {
            Some(state) => state,
            None => {
                search.ledger.retire(ticket).map_err(|retire| {
                    format!("unknown region; ticket retirement failed: {retire}")
                })?;
                return Err("unknown region".into());
            }
        };
        let pool = state_ref.pool_layout();
        let state = std::mem::replace(state_ref, state_ref.clone());
        let (state, proposal) =
            match state.ask_round(pool.arms() as usize, pool.slots() as usize, seed, config) {
                Ok(pair) => pair,
                Err(error) => {
                    search
                        .ledger
                        .retire(ticket)
                        .map_err(|retire| format!("{error}; ticket retirement failed: {retire}"))?;
                    return Err(error);
                }
            };
        search.states[slot] = state;
        search.pending_proposals[slot] = Some(proposal.clone());
        search.pending_tickets[slot] = Some(ticket);
        Ok((search, RegionalProposal { ticket, proposal }))
    }

    pub fn propose(
        self,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, RegionalProposal), String> {
        self.ask(seed, config)
    }

    pub fn propose_region(
        self,
        slot: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, RegionalProposal), String> {
        self.ask_region(slot, seed, config)
    }

    /// Select a restart center among `candidates` using `RestartCenter::First`
    /// or `RestartCenter::Maximin` (maximizing minimum GPU distance to
    /// active region centers).
    pub fn choose_center(&self, slot: usize, candidates: &[Buffer]) -> Result<usize, String> {
        if candidates.is_empty() {
            return Err("restart center candidates must be non-empty".into());
        }
        if self.restart_center == RestartCenter::First || candidates.len() == 1 || self.len() <= 1 {
            return Ok(0);
        }
        let mut best_idx = 0usize;
        let mut best_min_dist = f64::NEG_INFINITY;
        for (idx, cand) in candidates.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            for (other_slot, state) in self.states.iter().enumerate() {
                if other_slot == slot || state.history == 0 || state.async_command.is_some() {
                    continue;
                }
                let d = f64::from(state.measure_center(cand)?);
                min_dist = min_dist.min(d);
            }
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = idx;
            }
        }
        Ok(best_idx)
    }

    /// Deterministically generate `restart_candidates` candidate centers in
    /// `[0, 1]^D` and return the winning `u16` row under `RestartCenter`
    /// (`First` or `Maximin`).
    pub fn sample_center(&self, slot: usize, seed: u64) -> Result<Vec<u16>, String> {
        let ref_state = self.states.get(slot).ok_or("unknown region")?;
        let dims = ref_state.dimensions;
        let fp16 = ref_state.independent_fp16;
        let k = self.restart_candidates.max(1);
        let mut rows = Vec::with_capacity(k);
        let mut bufs = Vec::with_capacity(k);
        for c in 0..k {
            let root = crate::hash::splitmix64(
                seed ^ ((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    ^ ((c as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)),
            );
            let row: Vec<u16> = (0..dims)
                .map(|j| {
                    let h = crate::hash::splitmix64(
                        root.wrapping_add((j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                    );
                    let u = ((h >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
                    encode_unit(u, fp16)
                })
                .collect();
            if k > 1 && self.restart_center == RestartCenter::Maximin {
                bufs.push(ref_state.runtime.buffer_with(&row));
            }
            rows.push(row);
        }
        if bufs.is_empty() {
            return Ok(rows.swap_remove(0));
        }
        let chosen = self.choose_center(slot, &bufs)?;
        Ok(rows.swap_remove(chosen))
    }

    pub fn buffer(&self, proposal: &RegionalProposal) -> Result<Buffer, String> {
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.states
            .get(proposal.ticket.region.slot)
            .ok_or("unknown region")?
            .propose_buffer(&proposal.proposal)
    }

    pub fn materialize_batch(&self, proposal: &RegionalProposal) -> Result<Vec<DeviceRow>, String> {
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.states
            .get(proposal.ticket.region.slot)
            .ok_or("unknown region")?
            .materialize_batch(&proposal.proposal)
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
        self,
        proposal: &RegionalProposal,
        value: f32,
        variance: f32,
    ) -> Result<(Self, NoisyDecision), String> {
        self.tell_result(proposal, value, variance)
            .map_err(|failure| match failure {
                TellFailure::Retryable(message)
                | TellFailure::Retired(message)
                | TellFailure::Stale(message) => message,
            })
    }

    pub fn tell_result(
        self,
        proposal: &RegionalProposal,
        value: f32,
        variance: f32,
    ) -> Result<(Self, NoisyDecision), TellFailure> {
        tell::complete(
            self,
            proposal,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub fn tell_objectives(
        self,
        proposal: &RegionalProposal,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(Self, NoisyDecision), String> {
        objectives::tell(self, proposal, observation)
    }

    pub fn import_vec(
        self,
        slot: usize,
        row: &Buffer,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<Self, String> {
        objectives::import(self, slot, row, observation)
    }

    pub fn tell_arms(
        mut self,
        proposal: &RegionalProposal,
        rows: &[DeviceRow],
        obs: &[crate::objective_observation::ObjectiveObservation],
    ) -> Result<(Self, NoisyDecision), String> {
        use baux::semantics::optimizer::surrogate::BetterScore;
        if rows.is_empty() || rows.len() != obs.len() {
            return Err("tell_arms requires non-empty matching rows and observations".into());
        }
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        let slot = proposal.ticket.region.slot;
        if self.pending_tickets.get(slot).copied().flatten() != Some(proposal.ticket) {
            return Err("regional proposal is no longer pending in its generation".into());
        }
        for item in obs {
            item.validate()?;
        }

        let win = {
            let state = self.states.get(slot).ok_or("unknown region")?;
            let mut best_arm = 0usize;
            let mut best_key = (
                false,
                BetterScore(f64::NEG_INFINITY),
                std::cmp::Reverse(0usize),
            );
            for (arm, &item) in obs.iter().enumerate() {
                let (accepted, gain) = match state.objective_decision(item)? {
                    Some(pair) => pair,
                    None => {
                        let val = item.control().mean;
                        let acc = if state.implicit_history
                            && state.history >= state.initial_observations
                            && proposal.proposal.incumbent_mean.is_finite()
                        {
                            val > proposal.proposal.incumbent_mean
                        } else {
                            val > state.best
                        };
                        (acc, f64::from(val))
                    }
                };
                let key = (accepted, BetterScore(gain), std::cmp::Reverse(arm));
                if arm == 0 || key > best_key {
                    best_key = key;
                    best_arm = arm;
                }
            }
            best_arm
        };

        let mut win_prop = proposal.clone();
        if win > 0 && win < proposal.proposal.selected_indices.len() {
            let cand_idx = proposal.proposal.selected_indices[win];
            let win_buf = rows[win].wait()?;
            let state = self.states.get_mut(slot).ok_or("unknown region")?;
            propose::cow_buffer(&state.runtime, &mut state.proposal)?;
            state.copy(win_buf, &state.proposal)?;
            win_prop.proposal.index = cand_idx;
            if let Some(entry) = win_prop.proposal.pool.get(cand_idx) {
                win_prop.proposal.seed = entry.1;
                win_prop.proposal.length = entry.2;
                win_prop.proposal.history_distances = entry.4.clone();
            }
            if let Some(&(pm, pse, sc)) = win_prop.proposal.pool_predictions.get(cand_idx) {
                if pm.is_finite() {
                    win_prop.proposal.predicted_mean = pm;
                }
                if pse.is_finite() {
                    win_prop.proposal.predicted_standard_error = pse;
                }
                if sc.is_finite() {
                    win_prop.proposal.score = sc;
                }
            }
            let rad = win_prop
                .proposal
                .candidate_radii
                .get(cand_idx)
                .copied()
                .unwrap_or(0.0);
            win_prop.proposal.direction_norm = rad * rad;
            let root_seed = read::<Decision>(&state.decision, 1)[0].root_seed;
            win_prop.proposal.axis_coordinate = state.axis_candidate(root_seed, cand_idx);
            win_prop.proposal.axis_value = state
                .axis
                .as_ref()
                .zip(win_prop.proposal.axis_coordinate)
                .map(|(axis, coord)| axis.value(coord));
            win_prop.proposal.device_row =
                DeviceRow::new(Arc::clone(&state.proposal), None, state.dimensions);
            state.pending = Some(win_prop.proposal.clone());
            self.pending_proposals[slot] = Some(win_prop.proposal.clone());
        }

        let (mut self_next, dec) = self.tell_objectives(&win_prop, obs[win])?;

        for (arm, &item) in obs.iter().enumerate() {
            if arm == win {
                continue;
            }
            let buf = rows[arm].wait()?;
            if self_next.states[slot].has_surrogate() {
                self_next = objectives::import(self_next, slot, buf, item)?;
            }
            if self_next.restart_memory == RegionMemory::RetainRelevant {
                let st = &self_next.states[slot];
                let snap = st.runtime.buffer::<u16>(st.dimensions);
                st.copy(buf, &snap)?;
                if self_next.retained_archive.len() >= 2 * MAX_HISTORY {
                    self_next.retained_archive.remove(0);
                }
                self_next.retained_archive.push((Arc::new(snap), item));
            }
            if self_next.sharing != SharingPolicy::OriginOnly {
                let routes = routing::routes(&self_next, Some(slot), buf)?;
                if !routes.is_empty() {
                    let st = &self_next.states[slot];
                    let stable = st.runtime.buffer::<u16>(st.dimensions);
                    st.copy(buf, &stable)?;
                    for peer in routes {
                        if self_next.pending_imports[peer].len() < MAX_HISTORY {
                            self_next.pending_imports[peer].push(PendingImport {
                                row: stable.clone(),
                                observation: item,
                                from_region: true,
                            });
                        }
                    }
                }
            }
        }

        Ok((self_next, dec))
    }

    pub fn tell_batch(
        mut self,
        items: &[(RegionalProposal, f32, f32)],
    ) -> Result<(Self, Vec<NoisyDecision>), String> {
        let mut decisions = Vec::with_capacity(items.len());
        for (proposal, value, variance) in items {
            let (next_self, decision) = self.tell(proposal, *value, *variance)?;
            self = next_self;
            decisions.push(decision);
        }
        Ok((self, decisions))
    }

    pub fn batch_refs(
        mut self,
        items: &[(&RegionalProposal, f32, f32)],
    ) -> Result<(Self, Vec<NoisyDecision>), String> {
        let mut decisions = Vec::with_capacity(items.len());
        for &(proposal, value, variance) in items {
            let (next_self, decision) = self.tell(proposal, value, variance)?;
            self = next_self;
            decisions.push(decision);
        }
        Ok((self, decisions))
    }

    pub fn cancel(mut self, proposal: &RegionalProposal) -> Result<Self, String> {
        self.ledger
            .validate(proposal.ticket)
            .map_err(|error| error.to_string())?;
        let slot = proposal.ticket.region.slot;
        let state_ref = self.states.get_mut(slot).ok_or("unknown region")?;
        let state = std::mem::replace(state_ref, state_ref.clone());
        *state_ref = state.cancel_round(&proposal.proposal)?;
        self.ledger
            .retire(proposal.ticket)
            .map_err(|error| error.to_string())?;
        self.pending_proposals[proposal.ticket.region.slot] = None;
        self.pending_tickets[proposal.ticket.region.slot] = None;
        Ok(self)
    }

    pub fn import_region(
        self,
        slot: usize,
        row: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<Self, String> {
        objectives::import(
            self,
            slot,
            row,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
        )
    }

    pub fn import(
        self,
        slot: usize,
        row: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<Self, String> {
        self.import_region(slot, row, value, variance)
    }

    pub fn import_external(
        mut self,
        row: &Buffer,
        value: f32,
        variance: f32,
    ) -> Result<(Self, Vec<usize>), String> {
        check_scores(&[value], &[variance])?;
        if self.sharing == SharingPolicy::OriginOnly {
            return Err("origin-only sharing needs an explicit import_region target".into());
        }
        let routes = routing::routes(&self, None, row)?;
        for &slot in &routes {
            if self.pending_imports[slot].len() == MAX_HISTORY {
                return Err("resident sharing queue is full; flush a recipient".into());
            }
        }
        if routes.is_empty() {
            return Ok((self, routes));
        }
        let stable = {
            let state = self.states.get(routes[0]).ok_or("unknown region")?;
            let buf = state.runtime.buffer::<u16>(state.dimensions);
            state.copy(row, &buf)?;
            buf
        };
        for &slot in &routes {
            self.pending_imports[slot].push(PendingImport {
                row: stable.clone(),
                observation: crate::objective_observation::ObjectiveObservation::scalar_adapter(
                    value, variance,
                ),
                from_region: false,
            });
        }
        Ok((self, routes))
    }

    pub fn sharing_backlog(&self) -> Vec<usize> {
        self.pending_imports.iter().map(Vec::len).collect()
    }

    pub fn delivered_imports(&self) -> u64 {
        self.delivered_imports
    }

    pub fn flush_sharing(self, slot: usize) -> Result<Self, String> {
        routing::flush_region(self, slot, false)
    }

    pub fn restart(mut self, slot: usize, mut state: SearchState) -> Result<Self, String> {
        state.check_idle()?;
        let best = if state
            .incumbent_objectives()?
            .is_some_and(|record| record.observation.is_vector())
        {
            0.0
        } else {
            f64::from(state.best()?)
        };
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
            let old_ref = self.states.get_mut(slot).ok_or("unknown region")?;
            let old = std::mem::replace(old_ref, old_ref.clone());
            if let Ok(cancelled) = old.cancel_round(&old_proposal) {
                *old_ref = cancelled;
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
        if self.restart_memory == RegionMemory::RetainRelevant && state.has_surrogate() {
            for (arc_row, obs) in &self.retained_archive {
                if routing::state_relevant(&state, self.sharing, arc_row).unwrap_or(false) {
                    if let Ok(imported) = state.clone().import_objectives(arc_row, *obs) {
                        state = imported;
                    }
                }
            }
        }
        self.states[slot] = state;
        self.pending_imports[slot].clear();
        self.pending_tickets[slot] = None;
        Ok(self)
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
