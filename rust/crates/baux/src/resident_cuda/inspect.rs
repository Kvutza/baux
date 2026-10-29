use super::{Proposal, Proposals, SearchState, slot_count};

impl SearchState {
    /// Copy row zero into an unused, already allocated pending slot.
    pub fn snapshot_incumbent(&self) -> Result<(), String> {
        check_idle(self)?;
        self.engine.snapshot_incumbent(self.capacity + 1)
    }

    pub fn device_incumbent(&self, stream: Option<i64>) -> Result<(u64, usize, usize), String> {
        check_idle(self)?;
        let (pointer, _, _) = self.engine.device_row(self.capacity + 1, stream)?;
        Ok((pointer, 1, self.len()))
    }

    pub fn sync_consumers(&self) -> Result<(), String> {
        self.engine.sync_consumers()
    }

    pub fn device_row(
        &self,
        trial: Proposal,
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        let pending = self.pending_for(trial)?;
        self.engine.device_row(pending.slot, stream)
    }

    pub fn device_batch(
        &mut self,
        trials: &[Proposal],
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        if trials.is_empty() {
            return Err("BF16 device batch cannot be empty".to_string());
        }
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        self.engine.device_batch(&slots, stream)
    }

    pub fn device_round(
        &mut self,
        round: &Proposals,
        stream: Option<i64>,
    ) -> Result<(u64, usize, usize), String> {
        self.check_trials(&round.trials)?;
        self.engine.device_round(round.arms(), stream)
    }

    pub fn describe(
        &self,
        round: &Proposals,
    ) -> Result<Vec<baux_cuda::ProposalDescription>, String> {
        self.check_trials(&round.trials)?;
        self.engine.describe(round.arms())
    }

    pub fn set_tolerance(&mut self, tolerance: usize) -> Result<(), String> {
        if self.correlated {
            return Err("Correlated BF16 sampling does not use failure tolerance".into());
        }
        if tolerance == 0
            || u32::try_from(tolerance).is_err()
            || !self.pending.is_empty()
            || self.queued.is_some()
        {
            return Err(
                "BF16 failure tolerance must be a positive u32 set between completed rounds".into(),
            );
        }
        self.failure_limit = Some(tolerance);
        Ok(())
    }

    /// Selected candidate index and Gaussian persistence for each pending arm.
    /// Correlated indices 0/1 report 0.75; fresh and independent proposals
    /// report 0.
    pub fn geometry(&self, round: &Proposals) -> Result<Vec<(usize, f32)>, String> {
        self.check_trials(&round.trials)?;
        self.engine.geometry(round.arms())
    }

    pub fn read_best(&mut self) -> Result<Vec<u16>, String> {
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("Finish the pending BF16 round before exporting best weights".into());
        }
        self.engine.read(0)
    }

    /// Copy the stored correlated reference to host memory as raw BF16 words.
    /// This model-sized copy is for validation; other sampler modes return an
    /// error.
    pub fn read_reference(&mut self) -> Result<Vec<u16>, String> {
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("Finish the pending BF16 round before exporting the reference".into());
        }
        self.engine.read_reference()
    }

    pub fn read(&self, trial: Proposal) -> Result<Vec<u16>, String> {
        let pending = self.pending_for(trial)?;
        self.engine.read(pending.slot)
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.engine.set_profiling(enabled);
    }

    pub fn last_profile(&self) -> Option<baux_cuda::ProposalProfile> {
        self.engine.last_profile()
    }

    pub fn length(&mut self) -> Result<f64, String> {
        self.sync()?;
        Ok(self.length)
    }

    pub fn best(&mut self) -> Result<f32, String> {
        self.sync()?;
        Ok(self.best)
    }

    pub fn best_variance(&mut self) -> Result<f32, String> {
        self.sync()?;
        Ok(self.best_variance)
    }

    pub fn restarts(&mut self) -> Result<usize, String> {
        self.sync()?;
        Ok(self.restarts)
    }

    pub fn history_len(&mut self) -> Result<usize, String> {
        self.sync()?;
        Ok(self.history.len())
    }

    pub fn len(&self) -> usize {
        self.engine.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engine.is_empty()
    }

    pub(super) fn check_trials(&self, trials: &[Proposal]) -> Result<(), String> {
        if self.queued.is_some() {
            return Err("Sync the queued BF16 tell before another update".into());
        }
        for (index, trial) in trials.iter().enumerate() {
            if trials[..index].contains(trial) {
                return Err("BF16 tell batch contains a duplicate trial".to_string());
            }
            self.pending_for(*trial)?;
        }
        Ok(())
    }

    pub(super) fn pending_for(&self, trial: Proposal) -> Result<Proposal, String> {
        self.pending
            .iter()
            .copied()
            .find(|pending| pending.id == trial.id && *pending == trial)
            .ok_or_else(|| "BF16 trial does not match an outstanding ask".to_string())
    }

    pub(super) fn free_slots(&self, count: usize) -> Result<Vec<u32>, String> {
        let slots = (self.capacity + 1..slot_count(self.capacity, self.pending_capacity)?)
            .filter(|slot| self.pending.iter().all(|trial| trial.slot != *slot))
            .take(count)
            .map(|slot| slot as u32)
            .collect::<Vec<_>>();
        if slots.len() != count {
            Err("not enough free BF16 model slots".to_string())
        } else {
            Ok(slots)
        }
    }
}

fn check_idle(state: &SearchState) -> Result<(), String> {
    if !state.pending.is_empty() || state.queued.is_some() {
        return Err("Finish outstanding BF16 proposals and sync before exporting incumbent".into());
    }
    Ok(())
}
