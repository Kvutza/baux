//! Multi-tile 2-stage 512-dimensional H-Net execution for `BlockDecoder`.

use super::*;
use metal::MTLDispatchType;
use std::cell::RefCell;

impl BlockDecoder {
    fn make_scorer<'a>(
        &'a self,
        command: &'a CommandBufferRef,
        weights: CandidateRow<'a>,
        cache: Option<&'a [decode::Cache]>,
        row_start: u32,
    ) -> scorer::Scorer<'a> {
        scorer::Scorer {
            command,
            encoder: RefCell::new(Some(
                command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial),
            )),
            pipelines: &self.pipelines,
            tensorops: &self.tensorops,
            pisa1: &self.pisa,
            buffers: &self.buffers,
            weights,
            row_start,
            rows: self.buffers.rows,
            active_rows: std::cell::Cell::new(self.buffers.rows),
            cache,
            dirty: false,
            trace: None,
            feedback: crate::config::FeedbackTransition::Identity,
            diffusion: None,
            cache_only: false,
        }
    }

    fn prep_cmd(
        &self,
        runtime: &Runtime,
        weights: CandidateRow<'_>,
        tiles: &[u32],
    ) -> Result<f64, String> {
        let command = runtime.queue.new_command_buffer().to_owned();
        let scorer = self.make_scorer(&command, weights, None, 0);
        scorer.hnet2_prep()?;
        let mut tile_off = 0u32;
        for &tile_len in tiles {
            scorer.fwd_tile(tile_off, tile_len)?;
            tile_off += tile_len;
        }
        if let Some(enc) = scorer.encoder.borrow_mut().take() {
            enc.end_encoding();
        }
        command.commit();
        complete_committed(&command)
    }

    fn core_cmd(
        &self,
        _decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        m1: u32,
    ) -> Result<(u32, f64), String> {
        let runtime = Runtime::shared()?;
        let cap = self.buffers.rows;
        let total = m1.max(1);
        let stride = total.div_ceil(cap);
        let valid_rows = total.div_ceil(stride);
        let aligned_rows = valid_rows.max(1).div_ceil(128) * 128;
        let command = runtime.queue.new_command_buffer().to_owned();
        self.buffers.fine_grained.reset_routes(&command);
        let scorer = self.make_scorer(&command, weights, None, 0);
        scorer.hnet2_stage2(total, valid_rows, aligned_rows, stride)?;
        if let Some(enc) = scorer.encoder.borrow_mut().take() {
            enc.end_encoding();
        }
        command.commit();
        let sec = complete_committed(&command)?;
        Ok((aligned_rows, sec))
    }

    fn rev_cmd(
        &self,
        runtime: &Runtime,
        weights: CandidateRow<'_>,
        tiles: &[u32],
        score: u32,
    ) -> Result<f64, String> {
        let command = runtime.queue.new_command_buffer().to_owned();
        let scorer = self.make_scorer(&command, weights, None, 0);
        scorer.hnet2_dechunk()?;
        let mut tile_off = 0u32;
        for &tile_len in tiles {
            scorer.rev_tile(&self.proposals, tile_off, tile_len, score)?;
            tile_off += tile_len;
        }
        if let Some(enc) = scorer.encoder.borrow_mut().take() {
            enc.end_encoding();
        }
        command.commit();
        complete_committed(&command)
    }

    pub(in crate::fbt_moe) fn hnet2_tiled(
        &self,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        tiles: &[u32],
        score: u32,
    ) -> Result<f64, String> {
        let runtime = Runtime::shared()?;
        let hnet2 = self.buffers.hnet2()?;
        hnet2.reset_seq()?;
        let total_rows: u32 = tiles.iter().sum();
        let prep_sec = self.prep_cmd(&runtime, weights, tiles)?;
        let (_, m1) = hnet2.counts()?;
        let (aligned_m1, core_sec) = self.core_cmd(decoder, weights, m1)?;
        let rev_sec = self.rev_cmd(&runtime, weights, tiles, score)?;
        let total_sec = prep_sec + core_sec + rev_sec;
        hnet2.record_stats(total_rows, aligned_m1, prep_sec, core_sec, rev_sec)?;
        if self.tracing() {
            let stages = std::collections::BTreeMap::from([
                ("hnet2_prep", prep_sec * 1000.0),
                ("hnet2_core", core_sec * 1000.0),
                ("hnet2_rev", rev_sec * 1000.0),
            ]);
            self.finish_profile(0, total_rows, true, total_sec, stages)?;
        }
        Ok(total_sec)
    }

    pub(in crate::fbt_moe) fn score_hnet2(
        &self,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        total_rows: u32,
        score: u32,
    ) -> Result<f64, String> {
        let mut tiles = Vec::with_capacity((total_rows.div_ceil(HNET2_TILE)) as usize);
        let mut rem = total_rows;
        while rem > 0 {
            let step = rem.min(HNET2_TILE);
            tiles.push(step);
            rem -= step;
        }
        self.hnet2_tiled(decoder, weights, &tiles, score)
    }
}
