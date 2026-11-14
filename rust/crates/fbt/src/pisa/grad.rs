use crate::apple_gpu::thread_group;
use crate::fbt_pisa::{BATCH, HEAD_DIM, Pisa, QUERY_HEADS, QUERY_WIDTH};
use metal::{BufferRef, ComputeCommandEncoderRef};

impl Pisa {
    /// Differentiate selected attention entries in a packed [Q, K, V] prefix.
    /// Both the half primal and float adjoint use 640 elements per row; this
    /// method owns their different byte offsets into the shared storage.
    pub(crate) fn backward(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: (&BufferRef, u64),
        output_grad: (&BufferRef, u64),
        qkv_grad: (&BufferRef, u64),
        rows: u32,
    ) -> Result<(), String> {
        if rows == 0 || rows > self.context * BATCH {
            return Err("invalid PISA gradient shape".into());
        }
        let stride = QUERY_WIDTH + 2 * HEAD_DIM;
        for ((buffer, offset), width, bytes) in [
            (qkv, stride, 2u64),
            (qkv_grad, stride, 4),
            (output_grad, QUERY_WIDTH, 4),
        ] {
            if offset % bytes != 0
                || offset
                    .checked_add(u64::from(rows) * u64::from(width) * bytes)
                    .is_none_or(|end| end > buffer.length())
            {
                return Err("PISA adjoint slice exceeds its buffer".into());
            }
        }
        let shape = [0, rows, self.context, stride, stride];
        encoder.set_compute_pipeline_state(&self.pipelines.grad);
        for (slot, (buffer, offset)) in [
            qkv,
            (qkv.0, qkv.1 + u64::from(QUERY_WIDTH) * 2),
            (&self.blocks, 0),
            output_grad,
            qkv_grad,
            (qkv_grad.0, qkv_grad.1 + u64::from(QUERY_WIDTH) * 4),
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

#[cfg(test)]
#[test]
fn packed_adjoint() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        use crate::apple_gpu::{Runtime, buffer_slice, slice_mut};
        let runtime = Runtime::shared()?;
        let pisa = Pisa::with_context(&runtime, 4096)?;
        let blocks = slice_mut::<u32>(&pisa.blocks, 16)?;
        blocks.fill(u32::MAX);
        blocks[0] = 0;
        blocks[8] = 0;
        let mut packed = vec![0u16; 64 + 2 * 640 + 64];
        let mut output = vec![0.0f32; 2 * QUERY_WIDTH as usize];
        for head in 0..8 {
            packed[64 + 640 + head * 64] = 0x3c00;
            output[512 + head * 64] = 1.0;
        }
        packed[64 + 576] = 0x3c00;
        packed[64 + 640 + 576] = 0xbc00;
        let packed = runtime.buffer_with(&packed);
        let output = runtime.buffer_with(&output);
        let gradient = runtime.buffer_with(&vec![0.125f32; 64 + 2 * 640 + 64]);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        pisa.backward(&encoder, (&packed, 128), (&output, 0), (&gradient, 256), 2)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        let values = buffer_slice::<f32>(&gradient, 64 + 2 * 640 + 64)?;
        // The second query sees two zero keys and values +1/-1. Each of eight
        // heads contributes key adjoints +/-1/16 and value adjoints 1/2.
        for (index, &actual) in values.iter().enumerate() {
            let expected = match index {
                576 => 0.625,
                1216 => -0.375,
                640 | 1280 => 4.125,
                _ => 0.125,
            };
            assert_eq!(actual, expected, "packed adjoint index {index}");
        }
        Ok(())
    })
}
