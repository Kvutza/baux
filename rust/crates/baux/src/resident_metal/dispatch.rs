use super::*;
use crate::apple_gpu::thread_group;

impl SearchState {
    pub fn encode_proposal(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        let pipeline = if self.independent_fp16 && params.history == 1 && params.base_slot == 0 {
            &self.initial_pipeline
        } else {
            &self.propose_pipeline
        };
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in [
            &self.base,
            &self.anchor,
            &self.rejected,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub fn encode_pool(&self, command: &CommandBufferRef, params: Params) {
        for pair in 0..self.resident_history.div_ceil(2) {
            let first = pair * 2;
            let history = (self.resident_history - first).min(2);
            let base_slot = usize::try_from(params.base_slot)
                .ok()
                .and_then(|slot| slot.checked_sub(first))
                .filter(|&slot| slot < history)
                .map_or(2, |slot| slot as u32);
            let params = Params {
                history: history as u32,
                initialize: pair as u32,
                base_slot,
                ..params
            };
            self.encode_pair(command, params, first);
        }
    }
    pub fn encode_pair(&self, command: &CommandBufferRef, params: Params, first: usize) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pool_pipeline);
        for (index, buffer) in [
            &self.base,
            &self.history_rows[first],
            &self.history_rows[(first + 1).min(self.resident_history - 1)],
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub fn encode_select(
        &self,
        command: &CommandBufferRef,
        root: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) {
        let params = self.select_params(root, config, forced_candidate);
        if let Some(neighbors) = &self.neighbors {
            neighbors.encode(self, command, root);
        }
        let needs_reduction = forced_candidate.is_none() || self.family.is_some();
        if needs_reduction && !self.analytic_pool() {
            let reducer = command.new_compute_command_encoder();
            reducer.set_compute_pipeline_state(&self.reduction_pipeline);
            for (index, buffer) in [&self.partials, &self.pool_geometry, &self.pool_aggregates]
                .iter()
                .enumerate()
            {
                reducer.set_buffer(index as u64, Some(buffer), 0);
            }
            reducer.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            reducer.dispatch_thread_groups(
                thread_group(self.resident_history.div_ceil(2) as u64),
                thread_group(256),
            );
            reducer.end_encoding();
        }

        let encoder = command.new_compute_command_encoder();
        if let Some(selector) = &self.objective_selector {
            selector.upload(self, root, config);
            encoder.set_compute_pipeline_state(&selector.pipeline);
            encoder.set_buffer(4, Some(&selector.parameters), 0);
            encoder.set_buffer(5, Some(&selector.report), 0);
        } else {
            encoder.set_compute_pipeline_state(&self.selection_pipeline);
        }
        for (index, buffer) in [&self.pool_aggregates, &self.decision, &self.pool_distances]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            3,
            size_of::<SelectionParams>() as u64,
            (&params as *const SelectionParams).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();
    }
    pub fn encode_row(&self, command: &CommandBufferRef) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.materialize_pipeline);
        for (index, buffer) in [
            &self.base,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.decision,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        // Analytic pools select without materializing every candidate. Measure
        // the one row that is actually evaluated.
        let account = u32::from(self.analytic_pool());
        encoder.set_bytes(7, size_of::<u32>() as u64, (&account as *const u32).cast());
        encoder.set_buffer(8, Some(&self.partials), 0);
        let params = Params {
            tiles: self.tiles.len() as u32,
            ..self.pool_params(0)
        };
        encoder.set_bytes(
            9,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub fn encode_ref(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reference_pipeline);
        for (index, buffer) in [
            self.reference.as_ref().unwrap(),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.reference_partials,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            5,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_pipeline);
        for (index, buffer) in [
            &self.reference_partials,
            &self.leaves_gpu,
            &self.offsets_gpu,
            &self.reference_scales,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(thread_group(self.blocks.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub fn copy(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        autoreleasepool(|| self.copy_inner(source, destination))
    }

    pub fn copy_inner(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(source, 0, destination, 0, self.row_bytes());
        blit.end_encoding();
        finish(command)
    }
}
