use super::*;
use crate::apple_gpu::{barrier_buffers, thread_group};

fn grid_2d(width: u64, height: u64) -> metal::MTLSize {
    metal::MTLSize {
        width,
        height,
        depth: 1,
    }
}

impl SearchState {
    pub(super) fn meta_buffer(&self) -> &Buffer {
        self.manifold
            .as_ref()
            .map_or(&self.dummy_meta_gpu, |m| &*m.dim_meta_gpu)
    }

    pub(super) fn sobol_buffer(&self) -> &Buffer {
        self.sobol_dirs_gpu
            .as_ref()
            .map_or(&self.dummy_sobol_gpu, |b| &**b)
    }

    pub fn encode_proposal(&self, command: &CommandBufferRef, params: Params) {
        self.encode_into(command, params, &self.proposal);
    }

    pub(super) fn encode_into(&self, command: &CommandBufferRef, params: Params, output: &Buffer) {
        let encoder = command.new_compute_command_encoder();
        let pipeline = if self.independent_fp16 && params.history == 1 && params.base_slot == 0 {
            &self.initial_pipeline
        } else {
            &self.propose_pipeline
        };
        encoder.set_compute_pipeline_state(pipeline);
        let (first_row, second_row) = if self.history_rows.is_empty() {
            (&*self.base, &*self.base)
        } else {
            let second = self.resident_history.min(2).saturating_sub(1);
            (&*self.history_rows[0], &*self.history_rows[second])
        };
        for (index, buffer) in [
            &*self.base,
            first_row,
            second_row,
            &self.leaves_gpu,
            &self.tiles_gpu,
            output,
            &*self.partials,
            &*self.pool_geometry,
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
        encoder.set_buffer(9, Some(self.meta_buffer()), 0);
        encoder.set_buffer(10, Some(self.sobol_buffer()), 0);
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
        barrier_buffers(encoder);
        self.dispatch_select(encoder, seed, config, forced_candidate);
        if self.pool_layout.count() > 1 {
            barrier_buffers(encoder);
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
        if self.history_rows.is_empty() {
            let pair_params = Params {
                history: 1,
                initialize: 0,
                base_slot: 0,
                ..params
            };
            self.encode_buffers(encoder, pair_params, &self.base, &self.base);
            return;
        }
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
        let second = (first + 1).min(self.resident_history - 1);
        self.encode_buffers(
            encoder,
            params,
            &self.history_rows[first],
            &self.history_rows[second],
        );
    }

    fn encode_buffers(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        params: Params,
        first_row: &Buffer,
        second_row: &Buffer,
    ) {
        encoder.set_compute_pipeline_state(&self.pool_pipeline);
        for (index, buffer) in [
            &*self.base,
            first_row,
            second_row,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &*self.proposal,
            &*self.partials,
            &*self.pool_geometry,
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
        encoder.set_buffer(9, Some(self.meta_buffer()), 0);
        encoder.set_buffer(10, Some(self.sobol_buffer()), 0);
        let cand_groups = self.pool_layout.count().div_ceil(4).max(1) as u64;
        encoder.dispatch_thread_groups(
            grid_2d(self.tiles.len() as u64, cand_groups),
            thread_group(256),
        );
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
        self.upload_axis(root);
        let params = self.select_params(root, config, forced_candidate);
        let cand_groups = self.pool_layout.count().div_ceil(4).max(1) as u64;
        let pairs = self.resident_history.div_ceil(2).max(1) as u64;

        encoder.set_compute_pipeline_state(&self.reduction_pipeline);
        for (index, buffer) in [
            &*self.partials,
            &*self.pool_geometry,
            &*self.pool_aggregates,
        ]
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
        encoder.dispatch_thread_groups(grid_2d(pairs, cand_groups), thread_group(256));
        barrier_buffers(encoder);

        let score_tgs = self.pool_layout.count().div_ceil(256).max(1) as u64;
        if let Some(selector) = &self.objective_selector {
            selector.upload(self, root, config);
            encoder.set_compute_pipeline_state(&selector.score_pipeline);
            encoder.set_buffer(0, Some(&*self.pool_aggregates), 0);
            encoder.set_buffer(1, Some(&*self.pool_distances), 0);
            encoder.set_buffer(2, Some(&*self.candidate_scores), 0);
            encoder.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            encoder.set_buffer(4, Some(&*selector.parameters), 0);
            encoder.set_buffer(5, Some(&*selector.vectors), 0);
            encoder.set_buffer(6, Some(&*self.axis_candidates_gpu), 0);
            encoder.dispatch_thread_groups(thread_group(score_tgs), thread_group(256));
            barrier_buffers(encoder);

            encoder.set_compute_pipeline_state(&selector.pipeline);
            encoder.set_buffer(0, Some(&*self.candidate_scores), 0);
            encoder.set_buffer(1, Some(&*self.decision), 0);
            encoder.set_buffer(2, Some(&*self.selected_indices_gpu), 0);
            encoder.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            encoder.set_buffer(4, Some(&*selector.parameters), 0);
            encoder.set_buffer(5, Some(&*selector.report), 0);
            encoder.set_buffer(6, Some(&*selector.vectors), 0);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        } else {
            encoder.set_compute_pipeline_state(&self.score_pipeline);
            encoder.set_buffer(0, Some(&*self.pool_aggregates), 0);
            encoder.set_buffer(1, Some(&*self.pool_distances), 0);
            encoder.set_buffer(2, Some(&*self.candidate_scores), 0);
            encoder.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            encoder.set_buffer(4, Some(&*self.axis_candidates_gpu), 0);
            encoder.dispatch_thread_groups(thread_group(score_tgs), thread_group(256));
            barrier_buffers(encoder);

            encoder.set_compute_pipeline_state(&self.selection_pipeline);
            encoder.set_buffer(0, Some(&*self.candidate_scores), 0);
            encoder.set_buffer(1, Some(&*self.decision), 0);
            encoder.set_buffer(2, Some(&*self.selected_indices_gpu), 0);
            encoder.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        }
    }

    pub fn encode_row(&self, command: &CommandBufferRef) {
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_row(encoder);
        encoder.end_encoding();
    }

    pub fn dispatch_row(&self, encoder: &metal::ComputeCommandEncoderRef) {
        self.dispatch_candidate(encoder, &self.proposal, u32::MAX, 0, self.sobol_offset);
    }

    pub(super) fn dispatch_candidate(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        output: &Buffer,
        candidate_override: u32,
        account: u32,
        sobol_offset: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.materialize_pipeline);
        for (index, buffer) in [
            &*self.base,
            &self.leaves_gpu,
            &self.tiles_gpu,
            output,
            &*self.decision,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(5, size_of::<u32>() as u64, (&account as *const u32).cast());
        encoder.set_buffer(6, Some(&*self.partials), 0);
        let params = Params {
            tiles: self.tiles.len() as u32,
            candidate: candidate_override,
            sobol_offset,
            ..self.pool_params(0)
        };
        encoder.set_bytes(
            7,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.set_buffer(8, Some(self.meta_buffer()), 0);
        encoder.set_buffer(9, Some(self.sobol_buffer()), 0);
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
