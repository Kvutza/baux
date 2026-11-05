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
        let second_row = self.resident_history.min(2).saturating_sub(1);
        for (index, buffer) in [
            &self.base,
            &self.history_rows[0],
            &self.history_rows[second_row],
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
            8,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub fn encode_ask(
        &self,
        command: &CommandBufferRef,
        seed: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) {
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_pool(encoder, self.pool_params(seed));
        let pool_resources: [&metal::ResourceRef; 2] = [&self.partials, &self.pool_geometry];
        encoder.memory_barrier_with_resources(&pool_resources);
        self.dispatch_select(encoder, seed, config, forced_candidate);
        if self.pool_layout.count() > 1 {
            let select_resources: [&metal::ResourceRef; 2] = [&self.decision, &self.pool_distances];
            encoder.memory_barrier_with_resources(&select_resources);
            self.dispatch_row(encoder);
        }
        encoder.end_encoding();
    }
    pub fn encode_pool(&self, command: &CommandBufferRef, params: Params) {
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_pool(encoder, params);
        encoder.end_encoding();
    }
    pub fn dispatch_pool(&self, encoder: &metal::ComputeCommandEncoderRef, params: Params) {
        let pairs = self.resident_history.div_ceil(2);
        for pair in 0..pairs {
            let first = pair * 2;
            let history = (self.resident_history - first).min(2);
            let base_slot = usize::try_from(params.base_slot)
                .ok()
                .and_then(|slot| slot.checked_sub(first))
                .filter(|&slot| slot < history)
                .map_or(2, |slot| slot as u32);
            let pair_params = Params {
                history: history as u32,
                initialize: pair as u32,
                base_slot,
                ..params
            };
            self.encode_pair(encoder, pair_params, first);
        }
    }
    pub fn encode_pair(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        params: Params,
        first: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pool_pipeline);
        for (index, buffer) in [
            &self.base,
            &self.history_rows[first],
            &self.history_rows[(first + 1).min(self.resident_history - 1)],
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
            8,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
    }
    pub fn encode_select(
        &self,
        command: &CommandBufferRef,
        root: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) {
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_select(encoder, root, config, forced_candidate);
        encoder.end_encoding();
    }
    pub fn dispatch_select(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        root: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) {
        let params = self.select_params(root, config, forced_candidate);
        encoder.set_compute_pipeline_state(&self.reduction_pipeline);
        for (index, buffer) in [&self.partials, &self.pool_geometry, &self.pool_aggregates]
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
        encoder.dispatch_thread_groups(
            thread_group(self.resident_history.div_ceil(2).max(1) as u64),
            thread_group(256),
        );
        let resources: [&metal::ResourceRef; 2] = [&self.pool_aggregates, &self.pool_geometry];
        encoder.memory_barrier_with_resources(&resources);

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
    }
    pub fn encode_row(&self, command: &CommandBufferRef) {
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_row(encoder);
        encoder.end_encoding();
    }
    pub fn dispatch_row(&self, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.set_compute_pipeline_state(&self.materialize_pipeline);
        for (index, buffer) in [
            &self.base,
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
        let account = 0u32;
        encoder.set_bytes(5, size_of::<u32>() as u64, (&account as *const u32).cast());
        encoder.set_buffer(6, Some(&self.partials), 0);
        let params = Params {
            tiles: self.tiles.len() as u32,
            ..self.pool_params(0)
        };
        encoder.set_bytes(
            7,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
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
