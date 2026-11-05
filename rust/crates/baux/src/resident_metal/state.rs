use super::*;

impl SearchState {
    pub fn restarts(&self) -> Result<usize, String> {
        self.check_healthy()?;
        Ok(self.restart_count)
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }
    pub fn last_profile(&self) -> Option<ProposalProfile> {
        self.last_profile
    }
    pub fn tell_profile(&self) -> Option<TellProfile> {
        self.tell_profile
    }
    pub fn memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            row_bytes: self.row_bytes(),
            resident_bytes: self.resident_bytes,
            max_buffer_length: self.runtime.device.max_buffer_length(),
            recommended_max_working_set_size: self
                .runtime
                .device
                .recommended_max_working_set_size(),
            current_allocated_size: self.runtime.device.current_allocated_size(),
        }
    }

    pub fn controller_info(&self) -> Result<ControllerInfo, String> {
        self.check_healthy()?;
        if let Some(morbo) = self
            .objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
        {
            let trust = morbo.trust();
            return Ok(ControllerInfo {
                dimensions: self.dimensions,
                evaluated_arms: 1,
                length: trust.length(),
                length_min: self.length_config.length_min,
                length_max: self.length_config.length_max,
                success_tolerance: trust.success_tolerance(),
                failure_tolerance: trust.failure_tolerance(),
                success_counter: trust.success_counter(),
                failure_counter: trust.failure_counter(),
                restarts: self.restart_count,
            });
        }
        Ok(ControllerInfo {
            dimensions: self.dimensions,
            evaluated_arms: 1,
            length: self.length,
            length_min: self.length_config.length_min,
            length_max: self.length_config.length_max,
            success_tolerance: self.trust.success_tolerance(),
            failure_tolerance: self.trust.failure_tolerance(),
            success_counter: self.trust.success_counter(),
            failure_counter: self.trust.failure_counter(),
            restarts: self.restart_count,
        })
    }

    pub fn row_bytes(&self) -> u64 {
        self.dimensions as u64 * 2
    }
    pub fn check_healthy(&self) -> Result<(), String> {
        if self.poisoned {
            Err("Metal BF16 state is unusable after a failed GPU update".into())
        } else {
            Ok(())
        }
    }
    pub fn check_idle(&self) -> Result<(), String> {
        self.check_healthy()?;
        if self.pending.is_some() || self.queued.is_some() || self.async_command.is_some() {
            Err("Tell the pending Metal BF16 proposal and sync before another mutation".into())
        } else {
            Ok(())
        }
    }

    /// Retire a published proposal that did not produce an observation.
    /// Selection may have used scratch storage, but history and trust state
    /// have not been updated by a tell.
    pub fn cancel_round(&mut self, round: &Proposals) -> Result<(), String> {
        self.check_round(round)?;
        if self.queued.is_some() || self.async_command.is_some() {
            return Err("finish the pending Metal command before cancellation".into());
        }
        self.pending = None;
        self.evaluation = None;
        Ok(())
    }
}
