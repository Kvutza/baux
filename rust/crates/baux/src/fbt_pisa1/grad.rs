use crate::apple_gpu::thread_group;
use crate::fbt_pisa1::{BATCH, HEAD_DIM, Pisa1, QUERY_HEADS, QUERY_WIDTH};
use metal::{BufferRef, ComputeCommandEncoderRef};

impl Pisa1 {
    /// Apply the exact adjoint to the selected sparse-attention entries.
    pub(crate) fn backward(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: (&BufferRef, u64),
        kv: (&BufferRef, u64),
        output_grad: (&BufferRef, u64),
        query_grad: (&BufferRef, u64),
        kv_grad: (&BufferRef, u64),
        start: u32,
        rows: u32,
        query_stride: u32,
        kv_stride: u32,
    ) -> Result<(), String> {
        if rows == 0
            || start
                .checked_add(rows)
                .is_none_or(|end| end > self.context * BATCH)
            || query_stride < QUERY_WIDTH
            || kv_stride < 2 * HEAD_DIM
        {
            return Err("invalid PISA gradient shape".into());
        }
        let shape = [start, rows, self.context, query_stride, kv_stride];
        encoder.set_compute_pipeline_state(&self.pipelines.grad);
        for (slot, (buffer, offset)) in [
            query,
            kv,
            (&self.blocks, 0),
            output_grad,
            query_grad,
            kv_grad,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            6,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(rows * QUERY_HEADS)),
            thread_group(64),
        );
        Ok(())
    }
}
