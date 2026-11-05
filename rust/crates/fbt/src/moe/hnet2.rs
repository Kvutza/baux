//! Scratch buffers and two-stage compression telemetry for 2-stage 512-d H-Net.

use super::*;
use crate::apple_gpu::{buffer_slice, slice_mut};

pub(crate) const HNET2_MAX0: u32 = 360_448;
pub(crate) const HNET2_MAX1: u32 = 196_608;
pub(crate) const HNET2_TILE: u32 = 32_768;
pub(crate) const HNET2_BLKS: usize = 256;
pub(crate) const HNET2_PARTS: usize = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Hnet2Stats {
    pub(crate) total_bytes: u32,
    pub(crate) stage0_chunks: u32,
    pub(crate) stage1_chunks: u32,
    pub(crate) aligned_m1: u32,
    pub(crate) cap_c0: u32,
    pub(crate) cap_c1: u32,
    pub(crate) f0: f32,
    pub(crate) g0: f32,
    pub(crate) f1: f32,
    pub(crate) g1: f32,
    pub(crate) ratio_loss0: f32,
    pub(crate) ratio_loss1: f32,
    pub(crate) penalty0: f32,
    pub(crate) penalty1: f32,
    pub(crate) total_penalty: f32,
    pub(crate) prep_seconds: f64,
    pub(crate) core_seconds: f64,
    pub(crate) rev_seconds: f64,
}

pub(super) struct Hnet2Scratch {
    pub(super) waves: crate::hnet_metal::HNetWaves,
    pub(super) cap_bytes: u32,
    pub(super) cap_c0: u32,
    pub(super) cap_c1: u32,
    pub(super) mean_z: Buffer,
    pub(super) scalars: Buffer,
    pub(super) u0_table: Buffer,
    pub(super) gates_table: Buffer,
    pub(super) q0_table: Buffer,
    pub(super) k0_table: Buffer,
    pub(super) r0_table: Buffer,
    pub(super) q1_table: Buffer,
    pub(super) k1_table: Buffer,
    pub(super) r0_logits: Buffer,
    pub(super) b_logits: Buffer,
    pub(super) s0_blk_a: Buffer,
    pub(super) s0_blk_b: Buffer,
    pub(super) s1_blk_a: Buffer,
    pub(super) s1_blk_b: Buffer,
    pub(super) h0_state: Buffer,
    pub(super) q0_state: Buffer,
    pub(super) k0_state: Buffer,
    pub(super) q1s_state: Buffer,
    pub(super) k1s_state: Buffer,
    pub(super) h1_state: Buffer,
    pub(super) q1_state: Buffer,
    pub(super) k1_state: Buffer,
    pub(super) r_logit_state: Buffer,
    pub(super) y0_logit_state: Buffer,
    pub(super) count0: Buffer,
    pub(super) count1: Buffer,
    pub(super) tile_m0: Buffer,
    pub(super) tile_m1: Buffer,
    pub(super) seen0: Buffer,
    pub(super) seen1: Buffer,
    pub(super) tile_z0: Buffer,
    pub(super) tile_q1s: Buffer,
    pub(super) tile_k1s: Buffer,
    pub(super) probs0: Buffer,
    pub(super) plug0: Buffer,
    pub(super) origins0: Buffer,
    pub(super) probs1: Buffer,
    pub(super) plug1: Buffer,
    pub(super) origins1: Buffer,
    pub(super) chunks_h0: Buffer,
    pub(super) chunks_h1: Buffer,
    pub(super) core_h1: Buffer,
    pub(super) ema1: Buffer,
    pub(super) part_a: Buffer,
    pub(super) part_b: Buffer,
    pub(super) d0_part: Buffer,
    pub(super) tile_ema0_logits: Buffer,
    pub(super) ro_blk_abc: Buffer,
    pub(super) ro_blk_uv: Buffer,
    pub(super) last_stats: std::cell::Cell<Hnet2Stats>,
}

impl Hnet2Scratch {
    pub(super) fn new(
        runtime: &Runtime,
        waves: crate::hnet_metal::HNetWaves,
        context_cap: u32,
    ) -> Self {
        let cap_bytes = context_cap.max(4096);
        let (cap_c0, cap_c1) = if cap_bytes <= 4096 {
            (4096, 4096)
        } else {
            (HNET2_MAX0.min(cap_bytes), HNET2_MAX1.min(cap_bytes))
        };
        let d = WIDTH as usize;
        let v = 256usize;
        let tile = cap_bytes.min(HNET2_TILE) as usize;
        let blks = HNET2_BLKS;
        let parts = HNET2_PARTS;
        let c0 = cap_c0 as usize;
        let c1 = cap_c1 as usize;
        let cb = cap_bytes as usize;
        Self {
            waves,
            cap_bytes,
            cap_c0,
            cap_c1,
            mean_z: runtime.buffer::<f32>(d),
            scalars: runtime.buffer::<f32>(4),
            u0_table: runtime.buffer::<f32>(v * d),
            gates_table: runtime.buffer::<f32>(v * 2),
            q0_table: runtime.buffer::<f32>(v * d),
            k0_table: runtime.buffer::<f32>(v * d),
            r0_table: runtime.buffer::<f32>(v * d),
            q1_table: runtime.buffer::<f32>(v * d),
            k1_table: runtime.buffer::<f32>(v * d),
            r0_logits: runtime.buffer::<f32>(v * v),
            b_logits: runtime.buffer::<f32>(v),
            s0_blk_a: runtime.buffer::<f32>(1),
            s0_blk_b: runtime.buffer::<f32>(1),
            s1_blk_a: runtime.buffer::<f32>(blks),
            s1_blk_b: runtime.buffer::<f32>(blks * 3 * d),
            h0_state: runtime.buffer::<f32>(d),
            q0_state: runtime.buffer::<f32>(d),
            k0_state: runtime.buffer::<f32>(d),
            q1s_state: runtime.buffer::<f32>(d),
            k1s_state: runtime.buffer::<f32>(d),
            h1_state: runtime.buffer::<f32>(d),
            q1_state: runtime.buffer::<f32>(d),
            k1_state: runtime.buffer::<f32>(d),
            r_logit_state: runtime.buffer::<f32>(v),
            y0_logit_state: runtime.buffer::<f32>(v),
            count0: runtime.buffer_with(&[0u32]),
            count1: runtime.buffer_with(&[0u32]),
            tile_m0: runtime.buffer_with(&[0u32]),
            tile_m1: runtime.buffer_with(&[0u32]),
            seen0: runtime.buffer_with(&[0u32]),
            seen1: runtime.buffer_with(&[0u32]),
            tile_z0: runtime.buffer::<f32>(tile * d),
            tile_q1s: runtime.buffer::<f32>(tile * d),
            tile_k1s: runtime.buffer::<f32>(tile * d),
            probs0: runtime.buffer::<f32>(cb),
            plug0: runtime.buffer::<u32>(cb),
            origins0: runtime.buffer::<u32>(c0),
            probs1: runtime.buffer::<f32>(c0),
            plug1: runtime.buffer::<u32>(c0),
            origins1: runtime.buffer::<u32>(c1),
            chunks_h0: runtime.buffer::<f32>(c0 * d),
            chunks_h1: runtime.buffer::<u16>(c1 * d),
            core_h1: runtime.buffer::<u16>(c1 * d),
            ema1: runtime.buffer::<f32>(c1 * d),
            part_a: runtime.buffer::<f32>(1),
            part_b: runtime.buffer::<f32>(1),
            d0_part: runtime.buffer::<f32>(parts * 5 * d),
            tile_ema0_logits: runtime.buffer::<f32>((tile + 32) * v),
            ro_blk_abc: runtime.buffer::<f32>(1),
            ro_blk_uv: runtime.buffer::<f32>(1),
            last_stats: std::cell::Cell::new(Hnet2Stats::default()),
        }
    }

    pub(super) fn reset_seq(&self) -> Result<(), String> {
        let d = WIDTH as usize;
        let v = 256usize;
        for buf in [
            &self.h0_state,
            &self.q0_state,
            &self.k0_state,
            &self.q1s_state,
            &self.k1s_state,
            &self.h1_state,
            &self.q1_state,
            &self.k1_state,
        ] {
            slice_mut::<f32>(buf, d)?.fill(0.0);
        }
        slice_mut::<f32>(&self.r_logit_state, v)?.fill(0.0);
        slice_mut::<f32>(&self.y0_logit_state, v)?.fill(0.0);
        for buf in [
            &self.count0,
            &self.count1,
            &self.tile_m0,
            &self.tile_m1,
            &self.seen0,
            &self.seen1,
        ] {
            slice_mut::<u32>(buf, 1)?[0] = 0;
        }
        Ok(())
    }

    pub(super) fn counts(&self) -> Result<(u32, u32), String> {
        let m0 = buffer_slice::<u32>(&self.count0, 1)?[0].min(self.cap_c0);
        let m1 = buffer_slice::<u32>(&self.count1, 1)?[0].min(self.cap_c1);
        Ok((m0, m1))
    }

    pub(super) fn record_stats(
        &self,
        total_bytes: u32,
        aligned_m1: u32,
        prep_seconds: f64,
        core_seconds: f64,
        rev_seconds: f64,
    ) -> Result<Hnet2Stats, String> {
        let (m0, m1) = self.counts()?;
        let probs0 = buffer_slice::<f32>(&self.probs0, total_bytes as usize)?;
        let probs1 = buffer_slice::<f32>(&self.probs1, m0 as usize)?;
        let n0 = (total_bytes.max(1)) as f32;
        let d1 = (m0.max(1)) as f32;
        let f0 = (m0 as f32) / n0;
        let f1 = (m1 as f32) / d1;
        let (ratio_loss, total_penalty) =
            baux::semantics::hnet::hnet2_penalty(probs0, m0 as usize, probs1, m1 as usize);
        let stats = Hnet2Stats {
            total_bytes,
            stage0_chunks: m0,
            stage1_chunks: m1,
            aligned_m1,
            cap_c0: self.cap_c0,
            cap_c1: self.cap_c1,
            f0,
            g0: f0,
            f1,
            g1: f1,
            ratio_loss0: ratio_loss,
            ratio_loss1: 0.0,
            penalty0: total_penalty,
            penalty1: 0.0,
            total_penalty,
            prep_seconds,
            core_seconds,
            rev_seconds,
        };
        self.last_stats.set(stats);
        Ok(stats)
    }
}
