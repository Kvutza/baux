//! Chunking, EMA dechunking, and STE residual layers ported from H-Net.
//!
//! # Academic & Open-Source Attribution
//!
//! - **Paper**: Sukjun Hwang, Brandon Wang, and Albert Gu, *"Dynamic Chunking
//!   for End-to-End Hierarchical Sequence Modeling"*, arXiv:2507.07955 (2025).
//! - **Blog**: Cartesia AI, *"From Tokens to Bytes, Part 1"* (<https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1>).
//! - **Upstream Implementation**: Goomba Lab / Cartesia AI, <https://github.com/goombalab/hnet>
//!   (Apache-2.0 License, `hnet/modules/dc.py` and `hnet/models/hnet.py`:
//!   `ChunkLayer`, `DeChunkLayer`, `DeChunkState`, `STE`, `ste_func`,
//!   `residual_proj`, `pad_dimension`).

use super::EMA_EPS;

/// Padded batch of selected chunks (`hnet/modules/dc.py:ChunkLayer.forward`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkBatch {
    pub chunks: Vec<f32>,
    pub next_mask: Vec<bool>,
    pub num_chunks: Vec<usize>,
    pub batch_size: usize,
    pub max_chunks: usize,
    pub d_model: usize,
}

/// Packed batch of selected chunks (`hnet/modules/dc.py:ChunkLayer.forward`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPacked {
    pub chunks: Vec<f32>,
    pub cu_seqlens: Vec<usize>,
    pub max_seqlen: usize,
    pub d_model: usize,
}

/// Token-to-chunk compaction layer (`hnet/modules/dc.py:ChunkLayer`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkLayer;

impl ChunkLayer {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        boundary_mask: &[bool],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Result<ChunkBatch, String> {
        let total = batch_size * seq_len;
        if hidden_states.len() != total * d_model || boundary_mask.len() != total {
            return Err("ChunkLayer::forward shape mismatch".into());
        }
        let mut num_chunks = vec![0_usize; batch_size];
        let mut max_chunks = 0_usize;
        for (b, count) in num_chunks.iter_mut().enumerate() {
            let base = b * seq_len;
            *count = boundary_mask[base..base + seq_len]
                .iter()
                .filter(|&&m| m)
                .count();
            max_chunks = max_chunks.max(*count);
        }
        let mut chunks = vec![0.0_f32; batch_size * max_chunks * d_model];
        let mut next_mask = vec![false; batch_size * max_chunks];
        for b in 0..batch_size {
            let base = b * seq_len;
            let mut cursor = 0_usize;
            for t in 0..seq_len {
                if boundary_mask[base + t] {
                    let src = (base + t) * d_model;
                    let dst = (b * max_chunks + cursor) * d_model;
                    chunks[dst..dst + d_model].copy_from_slice(&hidden_states[src..src + d_model]);
                    next_mask[b * max_chunks + cursor] = true;
                    cursor += 1;
                }
            }
            for t in 0..seq_len {
                if cursor >= max_chunks {
                    break;
                }
                if !boundary_mask[base + t] {
                    let src = (base + t) * d_model;
                    let dst = (b * max_chunks + cursor) * d_model;
                    chunks[dst..dst + d_model].copy_from_slice(&hidden_states[src..src + d_model]);
                    cursor += 1;
                }
            }
        }
        Ok(ChunkBatch {
            chunks,
            next_mask,
            num_chunks,
            batch_size,
            max_chunks,
            d_model,
        })
    }

    pub fn forward_packed(
        &self,
        hidden_states: &[f32],
        boundary_mask: &[bool],
        cu_seqlens: &[usize],
        d_model: usize,
    ) -> Result<ChunkPacked, String> {
        let total = boundary_mask.len();
        if cu_seqlens.len() < 2 || hidden_states.len() != total * d_model {
            return Err("ChunkLayer::forward_packed shape mismatch".into());
        }
        let mut cumsum = Vec::with_capacity(total);
        let mut chunks = Vec::new();
        let mut running = 0_usize;
        for (t, &is_boundary) in boundary_mask.iter().enumerate() {
            if is_boundary {
                running += 1;
                let src = t * d_model;
                chunks.extend_from_slice(&hidden_states[src..src + d_model]);
            }
            cumsum.push(running);
        }
        let mut next_cu = Vec::with_capacity(cu_seqlens.len());
        next_cu.push(0);
        let mut max_seqlen = 0_usize;
        for window in cu_seqlens.windows(2) {
            let end = window[1];
            let next_end = if end == 0 { 0 } else { cumsum[end - 1] };
            let prev_end = *next_cu.last().unwrap_or(&0);
            max_seqlen = max_seqlen.max(next_end.saturating_sub(prev_end));
            next_cu.push(next_end);
        }
        Ok(ChunkPacked {
            chunks,
            cu_seqlens: next_cu,
            max_seqlen,
            d_model,
        })
    }

    pub fn step(&self, hidden_states: &[f32], boundary_mask: &[bool], d_model: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for (b, &is_boundary) in boundary_mask.iter().enumerate() {
            if is_boundary {
                let src = b * d_model;
                out.extend_from_slice(&hidden_states[src..src + d_model]);
            }
        }
        out
    }
}

/// Recurrent EMA state for [`DeChunkLayer`]
/// (`hnet/modules/dc.py:DeChunkState`).
#[derive(Clone, Debug, PartialEq)]
pub struct DeChunkState {
    pub last_value: Vec<f32>,
    pub batch_size: usize,
    pub d_model: usize,
}

impl DeChunkState {
    pub fn new(batch_size: usize, d_model: usize) -> Self {
        Self {
            last_value: vec![0.0_f32; batch_size * d_model],
            batch_size,
            d_model,
        }
    }
}

fn ema_scan(chunks: &[f32], probs: &[f32], num_chunks: usize, d_model: usize) -> Vec<f32> {
    let mut ema = vec![0.0_f32; num_chunks * d_model];
    let mut state = vec![0.0_f32; d_model];
    for i in 0..num_chunks {
        let p = probs[i];
        let one_minus_p = 1.0_f32 - p;
        let src = &chunks[i * d_model..(i + 1) * d_model];
        let dst = &mut ema[i * d_model..(i + 1) * d_model];
        for d in 0..d_model {
            state[d] = p * src[d] + one_minus_p * state[d];
            dst[d] = state[d];
        }
    }
    ema
}

/// EMA smoothing and plug-back dechunking layer
/// (`hnet/modules/dc.py:DeChunkLayer`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeChunkLayer {
    pub d_model: usize,
    pub eps: f32,
}

impl DeChunkLayer {
    pub const fn new(d_model: usize) -> Self {
        Self {
            d_model,
            eps: EMA_EPS,
        }
    }

    pub fn allocate_state(&self, batch_size: usize) -> DeChunkState {
        DeChunkState::new(batch_size, self.d_model)
    }

    pub fn forward(
        &self,
        chunk_states: &[f32],
        max_chunks: usize,
        boundary_mask: &[bool],
        boundary_prob: &[f32],
        batch_size: usize,
        seq_len: usize,
        mut state: Option<&mut DeChunkState>,
    ) -> Result<Vec<f32>, String> {
        let total = batch_size * seq_len;
        if boundary_mask.len() != total
            || boundary_prob.len() != total
            || chunk_states.len() != batch_size * max_chunks * self.d_model
        {
            return Err("DeChunkLayer::forward shape mismatch".into());
        }
        let mut out = vec![0.0_f32; total * self.d_model];
        for b in 0..batch_size {
            let base = b * seq_len;
            if state.is_some() && !boundary_mask[base] {
                return Err("First token must be a boundary if running prefill".into());
            }
            let mut probs = Vec::with_capacity(max_chunks);
            for t in 0..seq_len {
                if boundary_mask[base + t] {
                    probs.push(boundary_prob[base + t].clamp(self.eps, 1.0_f32 - self.eps));
                }
            }
            let count = probs.len();
            let seq_chunks = &chunk_states
                [b * max_chunks * self.d_model..(b * max_chunks + count) * self.d_model];
            let ema = ema_scan(seq_chunks, &probs, count, self.d_model);
            let mut cumsum = 0_usize;
            for t in 0..seq_len {
                if boundary_mask[base + t] {
                    cumsum += 1;
                }
                let plug_idx = cumsum.saturating_sub(1);
                let dst = (base + t) * self.d_model;
                if count > 0 {
                    let src = plug_idx * self.d_model;
                    out[dst..dst + self.d_model].copy_from_slice(&ema[src..src + self.d_model]);
                }
            }
            if let Some(st) = state.as_deref_mut() {
                let last_src = (base + seq_len - 1) * self.d_model;
                let dst = b * self.d_model;
                st.last_value[dst..dst + self.d_model]
                    .copy_from_slice(&out[last_src..last_src + self.d_model]);
            }
        }
        Ok(out)
    }

    pub fn forward_packed(
        &self,
        chunk_states: &[f32],
        boundary_mask: &[bool],
        boundary_prob: &[f32],
        next_cu_seqlens: &[usize],
    ) -> Result<Vec<f32>, String> {
        let total_tokens = boundary_mask.len();
        let total_chunks = *next_cu_seqlens.last().unwrap_or(&0);
        if boundary_prob.len() != total_tokens
            || chunk_states.len() != total_chunks * self.d_model
            || next_cu_seqlens.len() < 2
        {
            return Err("DeChunkLayer::forward_packed shape mismatch".into());
        }
        let mut probs = Vec::with_capacity(total_chunks);
        for (t, &is_boundary) in boundary_mask.iter().enumerate() {
            if is_boundary {
                probs.push(boundary_prob[t].clamp(self.eps, 1.0_f32 - self.eps));
            }
        }
        if probs.len() != total_chunks {
            return Err("Packed chunk count does not match boundary_mask sum".into());
        }
        let mut ema_all = vec![0.0_f32; total_chunks * self.d_model];
        for window in next_cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            let count = end.saturating_sub(start);
            let sub_chunks = &chunk_states[start * self.d_model..end * self.d_model];
            let sub_ema = ema_scan(sub_chunks, &probs[start..end], count, self.d_model);
            ema_all[start * self.d_model..end * self.d_model].copy_from_slice(&sub_ema);
        }
        let mut out = vec![0.0_f32; total_tokens * self.d_model];
        let mut cumsum = 0_usize;
        for (t, &is_boundary) in boundary_mask.iter().enumerate() {
            if is_boundary {
                cumsum += 1;
            }
            let plug_idx = cumsum.saturating_sub(1);
            let src = plug_idx * self.d_model;
            let dst = t * self.d_model;
            out[dst..dst + self.d_model].copy_from_slice(&ema_all[src..src + self.d_model]);
        }
        Ok(out)
    }

    pub fn step(
        &self,
        selected_chunks: &[f32],
        boundary_mask: &[bool],
        boundary_prob: &[f32],
        state: &mut DeChunkState,
    ) -> Result<Vec<f32>, String> {
        let batch_size = state.batch_size;
        let num_selected = boundary_mask.iter().filter(|&&m| m).count();
        if boundary_mask.len() != batch_size
            || boundary_prob.len() != batch_size
            || selected_chunks.len() != num_selected * self.d_model
            || state.d_model != self.d_model
        {
            return Err("DeChunkLayer::step shape mismatch".into());
        }
        let mut out = vec![0.0_f32; batch_size * self.d_model];
        let mut sel_cursor = 0_usize;
        for b in 0..batch_size {
            let state_slice = &mut state.last_value[b * self.d_model..(b + 1) * self.d_model];
            let out_slice = &mut out[b * self.d_model..(b + 1) * self.d_model];
            if boundary_mask[b] {
                let p = boundary_prob[b].clamp(self.eps, 1.0_f32 - self.eps);
                let one_minus_p = 1.0_f32 - p;
                let chunk =
                    &selected_chunks[sel_cursor * self.d_model..(sel_cursor + 1) * self.d_model];
                sel_cursor += 1;
                for d in 0..self.d_model {
                    let val = p * chunk[d] + one_minus_p * state_slice[d];
                    state_slice[d] = val;
                    out_slice[d] = val;
                }
            } else {
                out_slice.copy_from_slice(state_slice);
            }
        }
        Ok(out)
    }
}

/// Straight-Through Estimator residual combiner (`hnet/models/hnet.py:STE`,
/// `residual_proj`).
///
/// Forward: `out * ste_func(selected_probs) + residual_proj(encoder_out)`,
/// where `ste_func(p) = 1.0` in forward and `d(ste_func(p))/dp = 1.0` in
/// backward.
#[derive(Clone, Debug, PartialEq)]
pub struct SteResidual {
    pub d_model: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl SteResidual {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            weight: vec![0.0_f32; d_model * d_model],
            bias: vec![0.0_f32; d_model],
        }
    }

    pub fn forward(
        &self,
        dechunk_out: &[f32],
        encoder_out: &[f32],
        selected_probs: &[f32],
    ) -> Result<Vec<f32>, String> {
        let tokens = selected_probs.len();
        if dechunk_out.len() != tokens * self.d_model || encoder_out.len() != tokens * self.d_model
        {
            return Err("SteResidual::forward shape mismatch".into());
        }
        let mut combined = vec![0.0_f32; tokens * self.d_model];
        for t in 0..tokens {
            let enc = &encoder_out[t * self.d_model..(t + 1) * self.d_model];
            let dec = &dechunk_out[t * self.d_model..(t + 1) * self.d_model];
            let dst = &mut combined[t * self.d_model..(t + 1) * self.d_model];
            let ste_one = 1.0_f32 + 0.0_f32 * selected_probs[t];
            for row in 0..self.d_model {
                let w_row = &self.weight[row * self.d_model..(row + 1) * self.d_model];
                let mut res = self.bias[row];
                for (&w, &x) in w_row.iter().zip(enc.iter()) {
                    res += w * x;
                }
                dst[row] = dec[row] * ste_one + res;
            }
        }
        Ok(combined)
    }

    pub fn backward_prob(
        &self,
        grad_combined: &[f32],
        dechunk_out: &[f32],
        boundary_mask: &[bool],
    ) -> Result<Vec<f32>, String> {
        let tokens = boundary_mask.len();
        if grad_combined.len() != tokens * self.d_model
            || dechunk_out.len() != tokens * self.d_model
        {
            return Err("SteResidual::backward_prob shape mismatch".into());
        }
        let mut grad_p = vec![0.0_f32; tokens];
        for t in 0..tokens {
            let g = &grad_combined[t * self.d_model..(t + 1) * self.d_model];
            let y = &dechunk_out[t * self.d_model..(t + 1) * self.d_model];
            let mut dot = 0.0_f32;
            for (&gi, &yi) in g.iter().zip(y.iter()) {
                dot += gi * yi;
            }
            grad_p[t] = if boundary_mask[t] { dot } else { -dot };
        }
        Ok(grad_p)
    }
}
