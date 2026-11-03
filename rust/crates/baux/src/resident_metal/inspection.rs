use super::*;

impl SearchState {
    pub fn sync(&mut self) -> Result<Vec<bool>, String> {
        self.check_healthy()?;
        Ok(self.queued.take().into_iter().collect())
    }

    pub fn describe(&self, round: &Proposals) -> Result<Vec<ProposalDescription>, String> {
        self.check_round(round)?;
        let p = self.pending.as_ref().unwrap();
        Ok(vec![(p.seed, p.score, p.length, p.changes.clone())])
    }

    pub fn geometry(&self, round: &Proposals) -> Result<Vec<(usize, f32)>, String> {
        self.check_round(round)?;
        let index = self.pending.as_ref().unwrap().index;
        Ok(vec![(index, 0.0)])
    }

    pub fn base_id(&self, round: &Proposals) -> Result<i64, String> {
        self.check_round(round)?;
        Ok(round.base_id)
    }

    pub fn history_dists(&self, round: &Proposals) -> Result<Vec<(i64, f32)>, String> {
        self.check_round(round)?;
        Ok(round.history_distances.clone())
    }

    pub fn pool(&self, round: &Proposals) -> Result<Vec<PoolDescription>, String> {
        self.check_round(round)?;
        Ok(round.pool.clone())
    }

    pub fn pool_layout(&self) -> crate::procedural_pool::ProceduralPool {
        self.pool_layout
    }

    pub fn pool_geometry(&self, round: &Proposals) -> Result<PoolGeometry, String> {
        self.check_round(round)?;
        Ok((round.pool_radii, round.pool_cosines))
    }

    /// Current accepted BF16 weights, one contiguous row. Read-only lease
    /// required.
    pub fn base_buffer(&self) -> Buffer {
        self.base.clone()
    }
    pub fn best_buffer(&self) -> Buffer {
        self.base_buffer()
    }

    /// Selected BF16 row. A binding must keep its lease alive until GPU
    /// consumers finish.
    pub fn proposal_buffer(&self) -> Result<Buffer, String> {
        self.check_healthy()?;
        if self.pending.is_none() {
            return Err("No pending Metal BF16 proposal".into());
        }
        Ok(self.proposal.clone())
    }

    pub fn propose_buffer(&self, round: &Proposals) -> Result<Buffer, String> {
        self.check_round(round)?;
        self.proposal_buffer()
    }

    #[cfg(test)]
    pub fn test_candidate(&mut self, root: u64, index: usize) -> Result<Buffer, String> {
        self.check_idle()?;
        if index >= self.pool_layout.count() as usize {
            return Err("Metal BF16 candidate index out of pool bounds".into());
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_proposal(command, self.params(root, index));
        finish(command)?;
        Ok(self.proposal.clone())
    }

    #[cfg(test)]
    pub fn test_profile(&mut self, root: u64, config: RoundConfig) -> Result<[f32; 3], String> {
        self.check_idle()?;
        check_ask(config)?;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_pool(command, self.pool_params(root));
        finish(command)?;
        let pool_ms = start.elapsed().as_secs_f32() * 1000.0;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, None);
        finish(command)?;
        let select_ms = start.elapsed().as_secs_f32() * 1000.0;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_row(command);
        finish(command)?;
        let row_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok([pool_ms, select_ms, row_ms])
    }

    /// Validates a proposal handle before the binding exports its row.
    pub fn check_round(&self, round: &Proposals) -> Result<(), String> {
        self.check_healthy()?;
        if !self
            .pending
            .as_ref()
            .is_some_and(|p| p.owner == round.owner && p.id == round.id)
        {
            return Err("Stale or foreign Metal BF16 proposal".into());
        }
        Ok(())
    }

    /// Optional writable-consumer isolation using existing pending storage, no
    /// sixth row. The binding must invalidate this snapshot before ask and
    /// lease it while exported.
    pub fn snapshot(&self) -> Result<Buffer, String> {
        self.check_idle()?;
        self.copy(&self.base, &self.proposal)?;
        Ok(self.proposal.clone())
    }

    pub fn read_best(&self) -> Result<Vec<u16>, String> {
        self.check_healthy()?;
        Ok(read(&self.base, self.dimensions))
    }

    pub fn len(&self) -> usize {
        self.dimensions
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn length(&self) -> Result<f64, String> {
        self.check_healthy()?;
        Ok(self.length)
    }
    pub fn best(&self) -> Result<f32, String> {
        self.check_healthy()?;
        if self.history == 0 {
            return Err("Initial Metal BF16 incumbent has not been measured".into());
        }
        Ok(self.best)
    }
    pub fn best_variance(&self) -> Result<f32, String> {
        self.check_healthy()?;
        if self.history == 0 {
            return Err("Initial Metal BF16 incumbent has not been measured".into());
        }
        Ok(self.best_variance)
    }
    pub fn history_len(&self) -> Result<usize, String> {
        self.check_healthy()?;
        Ok(self.history)
    }
}

#[derive(Debug, Clone)]
pub struct MetalMemoryInfo {
    pub name: String,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_size: u64,
    pub current_allocated_size: u64,
    pub max_buffer_length: u64,
}

pub fn memory_info() -> Result<MetalMemoryInfo, String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        Ok(MetalMemoryInfo {
            name: runtime.device.name().to_owned(),
            has_unified_memory: runtime.device.has_unified_memory(),
            recommended_max_working_set_size: runtime.device.recommended_max_working_set_size(),
            current_allocated_size: runtime.device.current_allocated_size(),
            max_buffer_length: runtime.device.max_buffer_length(),
        })
    })
}

pub fn upload(weights: &[u16]) -> Result<Buffer, String> {
    autoreleasepool(|| {
        if weights.is_empty() || weights.iter().any(|x| x & 0x7f80 == 0x7f80) {
            return Err("Metal upload requires nonempty finite BF16 weights".into());
        }
        let bytes = weights
            .len()
            .checked_mul(2)
            .filter(|&x| x <= isize::MAX as usize)
            .ok_or("Metal allocation overflow")? as u64;
        let runtime = Runtime::shared()?;
        let recommended = runtime.device.recommended_max_working_set_size();
        let allocated = runtime.device.current_allocated_size();
        let limit = recommended - recommended / 10;
        if bytes > runtime.device.max_buffer_length()
            || allocated.checked_add(bytes).is_none_or(|x| x > limit)
        {
            return Err("Metal weight upload exceeds working set budget".into());
        }
        let buffer = runtime.buffer_with(weights);
        if buffer.contents().is_null() {
            return Err("Metal could not allocate weights".into());
        }
        Ok(buffer)
    })
}
