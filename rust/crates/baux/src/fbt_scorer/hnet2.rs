//! Metal compute dispatchers for the 2-stage 512-dimensional learned causal
//! H-Net.

use super::*;
use crate::apple_gpu::{bind_buffers, encode_value, thread_group};
use baux::semantics::hnet::HNetShape;
use metal::MTLSize;

fn size_2d(w: u64, h: u64) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: 1,
    }
}

impl Scorer<'_> {
    pub(super) fn quantize_layer(&self, layer: u32) -> Result<(), String> {
        if !self.pipelines.fine_grained.int8_gate() {
            return Ok(());
        }
        let pipeline = &self.pipelines.quantize_gate_up_int8;
        let output = self.buffers.quantized_gateup()?;
        let elements = routing::GATE_PARAMETERS as u64;
        let in_off = self.weights.gate_up + half_bytes(u64::from(layer) * elements);
        let out_off = u64::from(layer) * elements;
        let encoder = self.active_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.weights.buffer), in_off);
        encoder.set_buffer(1, Some(output), out_off);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&elements) as u64,
            (&elements as *const u64).cast(),
        );
        encoder.dispatch_threads(
            thread_group(elements),
            thread_group(pipeline.max_total_threads_per_threadgroup().min(256)),
        );
        encoder.memory_barrier_with_resources(&[output]);
        Ok(())
    }

    pub(in crate::fbt_moe) fn hnet2_prep(&self) -> Result<(), String> {
        self.quantize_layer(2)?;
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();

        enc.set_compute_pipeline_state(&p.prep_mean);
        bind_buffers(
            enc,
            &[
                (w.buffer, w.readout),
                (w.buffer, w.qkv),
                (w.buffer, w.output),
                (w.buffer, w.hnet),
                (w.buffer, w.mhc_attention_control),
                (w.buffer, w.mhc_moe_control),
                (&hnet2.mean_z, 0),
                (&hnet2.scalars, 0),
            ],
        );
        enc.dispatch_thread_groups(thread_group(1), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.mean_z, &hnet2.scalars]);

        enc.set_compute_pipeline_state(&p.prep_byte);
        bind_buffers(
            enc,
            &[
                (w.buffer, w.readout),
                (&hnet2.mean_z, 0),
                (w.buffer, w.router),
                (w.buffer, w.gate_up),
                (w.buffer, w.down),
                (w.buffer, w.attention_norm),
                (w.buffer, w.ffn_norm),
                (w.buffer, w.mhc_attention_control),
                (w.buffer, w.mhc_moe_control),
                (&hnet2.u0_table, 0),
                (&hnet2.gates_table, 0),
            ],
        );
        enc.dispatch_thread_groups(thread_group(256), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.u0_table, &hnet2.gates_table]);
        self.prep_tables()
    }

    fn prep_tables(&self) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();

        enc.set_compute_pipeline_state(&p.prep_proj);
        bind_buffers(
            enc,
            &[
                (&hnet2.u0_table, 0),
                (w.buffer, w.hnet),
                (w.buffer, w.attention_norm),
                (w.buffer, w.ffn_norm),
                (&hnet2.scalars, 0),
                (&hnet2.q0_table, 0),
                (&hnet2.k0_table, 0),
                (&hnet2.r0_table, 0),
                (&hnet2.q1_table, 0),
                (&hnet2.k1_table, 0),
            ],
        );
        enc.dispatch_threads(size_2d(512, 256), size_2d(256, 1));
        enc.memory_barrier_with_resources(&[
            &hnet2.q0_table,
            &hnet2.k0_table,
            &hnet2.r0_table,
            &hnet2.q1_table,
            &hnet2.k1_table,
        ]);

        enc.set_compute_pipeline_state(&p.prep_readout);
        bind_buffers(
            enc,
            &[
                (&hnet2.r0_table, 0),
                (w.buffer, w.hnet),
                (w.buffer, w.ffn_norm),
                (w.buffer, w.final_norm),
                (w.buffer, w.readout),
                (&hnet2.r0_logits, 0),
                (&hnet2.b_logits, 0),
            ],
        );
        enc.dispatch_threads(size_2d(256, 257), size_2d(256, 1));
        enc.memory_barrier_with_resources(&[&hnet2.r0_logits, &hnet2.b_logits]);
        Ok(())
    }

    pub(in crate::fbt_moe) fn fwd_tile(&self, tile_off: u32, tile_len: u32) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let shape = HNetShape {
            batch_size: tile_off,
            seq_len: tile_len,
            d_model: WIDTH,
            max_chunks: hnet2.cap_c0,
            vocab_size: hnet2.cap_c1,
            eps: 1e-4,
            target_ratio: 4.0,
            score: 1,
        };
        self.s0_tile(&shape)?;
        self.s1_tile(&shape)
    }

    fn advance_tile(&self, shape: &HNetShape, mode: u32) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let enc = self.active_encoder();
        let mut s = *shape;
        s.score = mode;
        enc.set_compute_pipeline_state(&p.tile_advance);
        bind_buffers(
            enc,
            &[
                (&hnet2.s0_blk_a, 0),
                (&hnet2.s0_blk_b, 0),
                (&hnet2.s1_blk_a, 0),
                (&hnet2.s1_blk_b, 0),
                (&hnet2.h0_state, 0),
                (&hnet2.q0_state, 0),
                (&hnet2.k0_state, 0),
                (&hnet2.q1s_state, 0),
                (&hnet2.k1s_state, 0),
                (&hnet2.h1_state, 0),
                (&hnet2.q1_state, 0),
                (&hnet2.k1_state, 0),
                (&hnet2.count0, 0),
                (&hnet2.tile_m0, 0),
                (&hnet2.count1, 0),
                (&hnet2.tile_m1, 0),
                (&hnet2.seen0, 0),
                (&hnet2.seen1, 0),
            ],
        );
        encode_value(enc, 18, &s);
        enc.dispatch_thread_groups(thread_group(1), thread_group(512));
        enc.memory_barrier_with_resources(&[
            &hnet2.s0_blk_b,
            &hnet2.s1_blk_b,
            &hnet2.h0_state,
            &hnet2.q0_state,
            &hnet2.k0_state,
            &hnet2.q1s_state,
            &hnet2.k1s_state,
            &hnet2.h1_state,
            &hnet2.q1_state,
            &hnet2.k1_state,
            &hnet2.count0,
            &hnet2.count1,
            &hnet2.seen0,
            &hnet2.seen1,
        ]);
        Ok(())
    }

    fn s0_tile(&self, shape: &HNetShape) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let enc = self.active_encoder();
        let n_blk = shape.seq_len.max(1).div_ceil(128).min(256) as u64;
        let n_route_blk = shape.seq_len.max(1).div_ceil(64).min(512) as u64;

        enc.set_compute_pipeline_state(&p.s0_route);
        bind_buffers(
            enc,
            &[
                (&self.buffers.tokens, 0),
                (&hnet2.q0_table, 0),
                (&hnet2.k0_table, 0),
                (&hnet2.gates_table, 0),
                (&hnet2.s0_blk_a, 0),
                (&hnet2.s0_blk_b, 0),
                (&hnet2.q0_state, 0),
                (&hnet2.k0_state, 0),
                (&hnet2.seen0, 0),
                (&hnet2.probs0, 0),
            ],
        );
        encode_value(enc, 10, shape);
        enc.dispatch_thread_groups(thread_group(n_route_blk.div_ceil(8)), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.probs0]);

        enc.set_compute_pipeline_state(&p.s0_scan);
        bind_buffers(
            enc,
            &[
                (&hnet2.probs0, 0),
                (&hnet2.count0, 0),
                (&hnet2.seen0, 0),
                (&hnet2.origins0, 0),
                (&hnet2.plug0, 0),
                (&hnet2.tile_m0, 0),
            ],
        );
        encode_value(enc, 6, shape);
        enc.dispatch_thread_groups(thread_group(1), thread_group(1024));
        enc.memory_barrier_with_resources(&[&hnet2.origins0, &hnet2.plug0, &hnet2.tile_m0]);

        enc.set_compute_pipeline_state(&p.s0_pack);
        bind_buffers(
            enc,
            &[
                (&self.buffers.tokens, 0),
                (&hnet2.u0_table, 0),
                (&hnet2.q1_table, 0),
                (&hnet2.k1_table, 0),
                (&hnet2.gates_table, 0),
                (&hnet2.s0_blk_a, 0),
                (&hnet2.s0_blk_b, 0),
                (&hnet2.h0_state, 0),
                (&hnet2.q1s_state, 0),
                (&hnet2.k1s_state, 0),
                (&hnet2.probs0, 0),
                (&hnet2.plug0, 0),
                (&hnet2.count0, 0),
                (&hnet2.seen0, 0),
                (&hnet2.tile_z0, 0),
                (&hnet2.tile_q1s, 0),
                (&hnet2.tile_k1s, 0),
            ],
        );
        encode_value(enc, 17, shape);
        enc.dispatch_threads(size_2d(512, n_blk), size_2d(256, 1));
        enc.memory_barrier_with_resources(&[&hnet2.tile_z0, &hnet2.tile_q1s, &hnet2.tile_k1s]);
        Ok(())
    }

    fn s1_tile(&self, shape: &HNetShape) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();
        let n_blk = shape.seq_len.max(1).div_ceil(128).min(256) as u64;

        enc.set_compute_pipeline_state(&p.s1_reduce);
        bind_buffers(
            enc,
            &[
                (&hnet2.tile_z0, 0),
                (&hnet2.tile_q1s, 0),
                (&hnet2.tile_k1s, 0),
                (w.buffer, w.attention_norm),
                (&hnet2.scalars, 0),
                (&hnet2.tile_m0, 0),
                (&hnet2.s1_blk_a, 0),
                (&hnet2.s1_blk_b, 0),
            ],
        );
        enc.dispatch_threads(size_2d(512, n_blk), size_2d(256, 1));
        enc.memory_barrier_with_resources(&[&hnet2.s1_blk_a, &hnet2.s1_blk_b]);
        self.advance_tile(shape, 1)?;

        enc.set_compute_pipeline_state(&p.s1_route);
        bind_buffers(
            enc,
            &[
                (&hnet2.tile_q1s, 0),
                (&hnet2.tile_k1s, 0),
                (&hnet2.scalars, 0),
                (&hnet2.s1_blk_a, 0),
                (&hnet2.s1_blk_b, 0),
                (&hnet2.q1_state, 0),
                (&hnet2.k1_state, 0),
                (&hnet2.count0, 0),
                (&hnet2.tile_m0, 0),
                (&hnet2.seen1, 0),
                (&hnet2.probs1, 0),
            ],
        );
        encode_value(enc, 11, shape);
        enc.dispatch_thread_groups(thread_group(n_blk.div_ceil(8)), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.probs1]);

        enc.set_compute_pipeline_state(&p.s1_scan);
        bind_buffers(
            enc,
            &[
                (&hnet2.probs1, 0),
                (&hnet2.count0, 0),
                (&hnet2.tile_m0, 0),
                (&hnet2.count1, 0),
                (&hnet2.seen1, 0),
                (&hnet2.origins1, 0),
                (&hnet2.plug1, 0),
                (&hnet2.tile_m1, 0),
            ],
        );
        encode_value(enc, 8, shape);
        enc.dispatch_thread_groups(thread_group(1), thread_group(1024));
        enc.memory_barrier_with_resources(&[&hnet2.origins1, &hnet2.plug1, &hnet2.tile_m1]);
        self.s1_finish(shape)
    }

    fn s1_finish(&self, shape: &HNetShape) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();
        let n_blk = shape.seq_len.max(1).div_ceil(128).min(256) as u64;

        enc.set_compute_pipeline_state(&p.s1_pack);
        bind_buffers(
            enc,
            &[
                (&hnet2.tile_z0, 0),
                (w.buffer, w.attention_norm),
                (&hnet2.scalars, 0),
                (&hnet2.s1_blk_a, 0),
                (&hnet2.s1_blk_b, 0),
                (&hnet2.h1_state, 0),
                (&hnet2.probs1, 0),
                (&hnet2.plug1, 0),
                (&hnet2.count0, 0),
                (&hnet2.tile_m0, 0),
                (&hnet2.seen1, 0),
                (&hnet2.chunks_h0, 0),
                (&hnet2.chunks_h1, 0),
            ],
        );
        encode_value(enc, 13, shape);
        enc.dispatch_threads(size_2d(512, n_blk), size_2d(256, 1));
        enc.memory_barrier_with_resources(&[&hnet2.chunks_h0, &hnet2.chunks_h1]);
        self.advance_tile(shape, 2)
    }

    pub(in crate::fbt_moe) fn hnet2_stage2(
        &self,
        total_m1: u32,
        valid_rows: u32,
        aligned_rows: u32,
        stride: u32,
    ) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let shape = HNetShape {
            batch_size: total_m1,
            seq_len: valid_rows,
            d_model: WIDTH,
            max_chunks: aligned_rows,
            vocab_size: stride,
            eps: 1e-4,
            target_ratio: 2.0,
            score: 1,
        };
        self.active_rows.set(aligned_rows);
        let enc = self.active_encoder();
        enc.set_compute_pipeline_state(&p.s2_load);
        bind_buffers(
            enc,
            &[(&hnet2.chunks_h1, 0), (&self.buffers.mhc_streams[0], 0)],
        );
        encode_value(enc, 2, &shape);
        enc.dispatch_threads(
            thread_group(u64::from(aligned_rows * WIDTH)),
            thread_group(256),
        );
        enc.memory_barrier_with_resources(&[&self.buffers.mhc_streams[0]]);

        self.mhc_attention(2, 2)?;
        self.mhc_moe(2)?;
        self.start_stage("mhc_readout")?;
        let normalized = self.activations(&self.buffers.normalized);
        let enc = self.active_encoder();
        self.hyper().finalize(enc, normalized.out());
        enc.memory_barrier_with_resources(&[&self.buffers.normalized]);

        enc.set_compute_pipeline_state(&p.s2_store);
        bind_buffers(
            enc,
            &[
                (&self.buffers.normalized, 0),
                (&hnet2.chunks_h1, 0),
                (&hnet2.core_h1, 0),
            ],
        );
        encode_value(enc, 3, &shape);
        enc.dispatch_threads(thread_group(u64::from(total_m1 * WIDTH)), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.core_h1]);
        Ok(())
    }

    pub(in crate::fbt_moe) fn hnet2_dechunk(&self) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();
        let shape = HNetShape {
            batch_size: 1,
            seq_len: self.rows,
            d_model: WIDTH,
            max_chunks: hnet2.cap_c0,
            vocab_size: hnet2.cap_c1,
            eps: 1e-4,
            target_ratio: 2.0,
            score: 1,
        };
        let grid = size_2d(u64::from(WIDTH), HNET2_PARTS as u64);
        let tg = size_2d(256, 1);

        enc.set_compute_pipeline_state(&p.dec1_scan);
        bind_buffers(
            enc,
            &[
                (&hnet2.core_h1, 0),
                (&hnet2.ema1, 0),
                (&hnet2.origins1, 0),
                (&hnet2.probs1, 0),
                (&hnet2.count1, 0),
                (&hnet2.part_a, 0),
                (&hnet2.part_b, 0),
            ],
        );
        encode_value(enc, 7, &shape);
        enc.dispatch_threads(grid, tg);
        enc.memory_barrier_with_resources(&[&hnet2.ema1]);

        enc.set_compute_pipeline_state(&p.dec0_reduce);
        bind_buffers(
            enc,
            &[
                (&hnet2.ema1, 0),
                (&hnet2.chunks_h0, 0),
                (&hnet2.plug1, 0),
                (&hnet2.origins0, 0),
                (&hnet2.probs0, 0),
                (&hnet2.count0, 0),
                (w.buffer, w.attention_norm),
                (w.buffer, w.ffn_norm),
                (&hnet2.scalars, 0),
                (&hnet2.d0_part, 0),
            ],
        );
        encode_value(enc, 10, &shape);
        enc.dispatch_threads(grid, tg);
        enc.memory_barrier_with_resources(&[&hnet2.d0_part]);

        enc.set_compute_pipeline_state(&p.dec0_scan);
        bind_buffers(
            enc,
            &[
                (&hnet2.ema1, 0),
                (&hnet2.chunks_h0, 0),
                (&hnet2.plug1, 0),
                (&hnet2.origins0, 0),
                (&hnet2.probs0, 0),
                (&hnet2.count0, 0),
                (w.buffer, w.attention_norm),
                (w.buffer, w.ffn_norm),
                (&hnet2.scalars, 0),
                (&hnet2.d0_part, 0),
                (&hnet2.tile_z0, 0),
            ],
        );
        encode_value(enc, 11, &shape);
        enc.dispatch_threads(grid, tg);
        enc.memory_barrier_with_resources(&[&hnet2.chunks_h0, &hnet2.tile_z0]);
        Ok(())
    }

    pub(in crate::fbt_moe) fn rev_tile(
        &self,
        proposals: &BufferRef,
        tile_off: u32,
        tile_len: u32,
        score: u32,
    ) -> Result<(), String> {
        let hnet2 = self.buffers.hnet2()?;
        let p = &hnet2.waves.hnet2;
        let w = self.weights;
        let enc = self.active_encoder();
        let shape = HNetShape {
            batch_size: tile_off,
            seq_len: tile_len,
            d_model: WIDTH,
            max_chunks: hnet2.cap_c0,
            vocab_size: hnet2.cap_c1,
            eps: 1e-4,
            target_ratio: 4.0,
            score,
        };

        let n_blk = tile_len.max(1).div_ceil(64).min(512) as u64;

        enc.set_compute_pipeline_state(&p.ema0_logits);
        bind_buffers(
            enc,
            &[
                (&hnet2.chunks_h0, 0),
                (&hnet2.plug0, 0),
                (w.buffer, w.ffn_norm),
                (w.buffer, w.final_norm),
                (w.buffer, w.readout),
                (&hnet2.b_logits, 0),
                (&hnet2.tile_ema0_logits, 0),
            ],
        );
        encode_value(enc, 7, &shape);
        enc.dispatch_thread_groups(thread_group(256), thread_group(256));
        enc.memory_barrier_with_resources(&[&hnet2.tile_ema0_logits]);

        enc.set_compute_pipeline_state(&p.ro_scan);
        bind_buffers(
            enc,
            &[
                (&self.buffers.tokens, 0),
                (&self.buffers.labels, 0),
                (&hnet2.plug0, 0),
                (&hnet2.gates_table, 0),
                (&hnet2.r0_logits, 0),
                (&hnet2.b_logits, 0),
                (&hnet2.tile_ema0_logits, 0),
                (&hnet2.ro_blk_abc, 0),
                (&hnet2.ro_blk_uv, 0),
                (&hnet2.r_logit_state, 0),
                (&hnet2.y0_logit_state, 0),
                (proposals, 0),
                (&self.buffers.losses, 0),
            ],
        );
        encode_value(enc, 13, &shape);
        enc.dispatch_thread_groups(thread_group(n_blk.div_ceil(8)), thread_group(256));
        enc.memory_barrier_with_resources(&[proposals, &self.buffers.losses]);
        Ok(())
    }
}
