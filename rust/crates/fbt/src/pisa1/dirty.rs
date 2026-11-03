use super::*;

impl Pisa1 {
    pub(crate) fn pyramid_range(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        pyramid: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        debug_assert!(rows > 0);
        debug_assert!(start + rows <= self.context);
        let first = start / BLOCK;
        let end = (start + rows).div_ceil(BLOCK);
        let range = [first, end - first, 0, 0];
        encoder.set_compute_pipeline_state(&self.pipelines.leaf_range);
        encoder.set_buffer(0, Some(qkv), 0);
        encoder.set_buffer(1, Some(pyramid), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(end - first)),
            thread_group(u64::from(HEAD_DIM)),
        );
        encoder.memory_barrier_with_resources(&[pyramid]);
        encoder.set_compute_pipeline_state(&self.pipelines.upper_range);
        encoder.set_buffer(0, Some(pyramid), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(u64::from(HEAD_DIM)));
        encoder.memory_barrier_with_resources(&[pyramid]);
    }

    pub(crate) fn attention_from(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        pyramid: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        debug_assert!(rows > 0);
        debug_assert!(start + rows <= self.query_rows);
        debug_assert_eq!(start % self.pipelines.query_tile, 0);
        debug_assert_eq!(rows % self.pipelines.query_tile, 0);
        let range = [start, rows];

        encoder.set_compute_pipeline_state(&self.pipelines.full_clear);
        encoder.set_buffer(0, Some(&self.leaf_counts), 0);
        encoder.set_buffer(1, Some(&self.leaf_tile_count), 0);
        encoder.dispatch_thread_groups(
            thread_group(u64::from((2 * self.context / BLOCK).div_ceil(128))),
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&self.leaf_counts, &self.leaf_tile_count]);

        encoder.set_compute_pipeline_state(&self.pipelines.full_intermediate);
        for (index, buffer) in [
            qkv,
            pyramid,
            &self.leaf_candidates,
            &self.leaf_counts,
            &self.leaf_refs,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(rows)), thread_group(32));
        encoder.memory_barrier_with_resources(&[
            &self.leaf_candidates,
            &self.leaf_counts,
            &self.leaf_refs,
        ]);

        encoder.set_compute_pipeline_state(&self.pipelines.full_tiles);
        encoder.set_buffer(0, Some(&self.leaf_counts), 0);
        encoder.set_buffer(1, Some(&self.leaf_tile_count), 0);
        encoder.set_buffer(2, Some(&self.leaf_tiles), 0);
        encoder.dispatch_thread_groups(
            thread_group(u64::from((2 * self.context / BLOCK).div_ceil(128))),
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&self.leaf_tiles, &self.leaf_tile_count]);

        encoder.set_compute_pipeline_state(&self.pipelines.full_leaf_q4);
        for (index, buffer) in [
            qkv,
            &self.leaf_refs,
            &self.leaf_counts,
            &self.leaf_tiles,
            &self.leaf_tile_count,
            &self.leaf_scores,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.leaf_tile_capacity)),
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&self.leaf_scores]);

        encoder.set_compute_pipeline_state(&self.pipelines.full_final);
        encoder.set_buffer(0, Some(&self.leaf_candidates), 0);
        encoder.set_buffer(1, Some(&self.leaf_scores), 0);
        encoder.set_buffer(2, Some(&self.blocks), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(rows)), thread_group(32));
        encoder.memory_barrier_with_resources(&[&self.blocks]);

        encoder.set_compute_pipeline_state(&self.pipelines.full_attention_q4);
        for (index, buffer) in [qkv, pyramid, &self.blocks, &self.output]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(rows / self.pipelines.query_tile)),
            thread_group(u64::from(self.pipelines.query_tile * 32)),
        );
        encoder.memory_barrier_with_resources(&[&self.output]);
    }

    fn encode_attention(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        let encoder = command.new_compute_command_encoder();
        self.attention_range(&encoder, qkv, 0, ROWS);
        encoder.end_encoding();
    }

    pub(super) fn encode(&self, command: &CommandBufferRef, qkv: &BufferRef, stage: Option<Stage>) {
        if stage.is_none_or(|value| matches!(value, Stage::Pyramid)) {
            self.encode_pyramid(command, qkv);
        }
        if matches!(stage, Some(Stage::Selection)) {
            self.encode_selection(command, qkv);
        }
        if stage.is_none_or(|value| matches!(value, Stage::Attention)) {
            self.encode_attention(command, qkv);
        }
    }

    pub(crate) fn encode_layer(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        self.encode(command, qkv, None);
    }

    pub(crate) fn pyramid_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        rows: u32,
    ) {
        if rows < self.context {
            self.pyramid_range(encoder, qkv, &self.pyramid, 0, rows);
        } else {
            self.pyramid_output(encoder, qkv, &self.pyramid, rows);
        }
    }

    pub(crate) fn attention_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        rows: u32,
    ) {
        self.attention_range(encoder, qkv, 0, rows);
    }

    pub(crate) fn attention_range(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        self.attention_from(encoder, qkv, &self.pyramid, start, rows);
    }
}
