//! Dynamic chunking and byte-level hierarchy semantics ported from H-Net.
//!
//! # Academic & Open-Source Attribution
//!
//! This module contains a Rust port of the dynamic chunking routing,
//! chunking, EMA dechunking, byte tokenizer, and load-balancing ratio loss from
//! H-Net:
//!
//! - **Paper**: Sukjun Hwang, Brandon Wang, and Albert Gu, *"Dynamic Chunking
//!   for End-to-End Hierarchical Sequence Modeling"*, arXiv:2507.07955 (2025).
//! - **Blog**: Cartesia AI, *"From Tokens to Bytes, Part 1"* (<https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1>).
//! - **Upstream Implementation**: Goomba Lab / Cartesia AI, <https://github.com/goombalab/hnet>
//!   (MIT License), specifically:
//!   - `hnet/modules/dc.py` (`RoutingModule`, `RoutingModuleOutput`,
//!     `RoutingModuleState`, `ChunkLayer`, `DeChunkLayer`, `DeChunkState`)
//!   - `hnet/models/hnet.py` (`STE`, `ste_func`, `HNet`, `residual_proj`)
//!   - `hnet/utils/tokenizers.py` (`ByteTokenizer`)
//!   - `hnet/utils/train.py` (`load_balancing_loss`)

use baux::semantics::abi::{metal_field, metal_kind, record};
#[path = "hnet/dechunk.rs"]
mod dechunk;
#[path = "hnet/stage2.rs"]
mod stage2;
pub use dechunk::{ChunkBatch, ChunkLayer, ChunkPacked, DeChunkLayer, DeChunkState, SteResidual};
pub use stage2::{
    HNet2Forward, HNet2Params, HNet2Reverse, hnet2_forward, hnet2_penalty, hnet2_reverse,
};

pub const BYTE_VOCAB: usize = 256;
pub const BOS_BYTE: u8 = 254;
pub const EOS_BYTE: u8 = 255;
pub const PAD_PROB: f32 = 1.0;
pub const EMA_EPS: f32 = 1e-4;
pub const NORM_EPS: f32 = 1e-12;

record! {
    HNetShape {
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        max_chunks: u32,
        vocab_size: u32,
        eps: f32,
        target_ratio: f32,
        score: u32,
    }
}

/// Preamble compiled with the consuming H-Net Metal kernel, including layout
/// checks.
pub fn metal_source(body: &str) -> String {
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(&HNetShape::metal_decl());
    source.push_str(body);
    source
}

/// UTF-8 byte-level tokenizer (`hnet/utils/tokenizers.py:ByteTokenizer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteTokenizer {
    pub vocab_size: usize,
    pub bos_byte: u8,
    pub eos_byte: u8,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteTokenizer {
    pub const fn new() -> Self {
        Self {
            vocab_size: BYTE_VOCAB,
            bos_byte: BOS_BYTE,
            eos_byte: EOS_BYTE,
        }
    }

    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u8> {
        let raw = text.as_bytes();
        let mut out = Vec::with_capacity(raw.len() + usize::from(add_bos) + usize::from(add_eos));
        if add_bos {
            out.push(self.bos_byte);
        }
        out.extend_from_slice(raw);
        if add_eos {
            out.push(self.eos_byte);
        }
        out
    }

    pub fn decode(&self, tokens: &[u8]) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(tokens.to_vec())
    }
}

/// Recurrent state for [`RoutingModule`]
/// (`hnet/modules/dc.py:RoutingModuleState`).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingState {
    pub has_seen: Vec<bool>,
    pub last_hidden: Vec<f32>,
    pub batch_size: usize,
    pub d_model: usize,
}

impl RoutingState {
    pub fn new(batch_size: usize, d_model: usize) -> Self {
        Self {
            has_seen: vec![false; batch_size],
            last_hidden: vec![0.0; batch_size * d_model],
            batch_size,
            d_model,
        }
    }
}

/// Output of [`RoutingModule`] (`hnet/modules/dc.py:RoutingModuleOutput`).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingOutput {
    pub boundary_prob: Vec<f32>,
    pub boundary_mask: Vec<bool>,
    pub selected_probs: Vec<f32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl RoutingOutput {
    pub fn prob_pair(&self, index: usize) -> [f32; 2] {
        let p = self.boundary_prob[index];
        [1.0 - p, p]
    }
}

fn project_norm(weight: &[f32], input: &[f32], d_model: usize) -> Vec<f32> {
    let mut proj = vec![0.0_f32; d_model];
    let mut norm_sq = 0.0_f32;
    for (row, out_val) in proj.iter_mut().enumerate() {
        let row_slice = &weight[row * d_model..(row + 1) * d_model];
        let mut sum = 0.0_f32;
        for (w, &x) in row_slice.iter().zip(input.iter()) {
            sum += *w * x;
        }
        *out_val = sum;
        norm_sq += sum * sum;
    }
    let inv_norm = 1.0_f32 / norm_sq.sqrt().max(NORM_EPS);
    for val in &mut proj {
        *val *= inv_norm;
    }
    proj
}

fn cosine_prob(norm_q: &[f32], norm_k: &[f32]) -> f32 {
    let mut cos_sim = 0.0_f32;
    for (&q, &k) in norm_q.iter().zip(norm_k.iter()) {
        cos_sim += q * k;
    }
    ((1.0_f32 - cos_sim) * 0.5_f32).clamp(0.0_f32, 1.0_f32)
}

/// Cosine-similarity boundary router (`hnet/modules/dc.py:RoutingModule`).
///
/// Both `q_proj` and `k_proj` are bias-free `[d_model, d_model]` linear layers
/// initialized to the identity matrix (`_no_reinit = True` in upstream).
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingModule {
    pub d_model: usize,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
}

impl RoutingModule {
    pub fn new(d_model: usize) -> Self {
        let mut eye = vec![0.0_f32; d_model * d_model];
        for i in 0..d_model {
            eye[i * d_model + i] = 1.0_f32;
        }
        Self {
            d_model,
            q_proj: eye.clone(),
            k_proj: eye,
        }
    }

    pub fn with_weights(
        d_model: usize,
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
    ) -> Result<Self, String> {
        let expected = d_model * d_model;
        if d_model == 0 || q_proj.len() != expected || k_proj.len() != expected {
            return Err(
                "RoutingModule requires non-empty d_model * d_model projection matrices".into(),
            );
        }
        Ok(Self {
            d_model,
            q_proj,
            k_proj,
        })
    }

    pub fn allocate_state(&self, batch_size: usize) -> RoutingState {
        RoutingState::new(batch_size, self.d_model)
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        batch_size: usize,
        seq_len: usize,
        mask: Option<&[bool]>,
        mut state: Option<&mut RoutingState>,
    ) -> Result<RoutingOutput, String> {
        let total = batch_size * seq_len;
        if batch_size == 0 || seq_len == 0 || hidden_states.len() != total * self.d_model {
            return Err("RoutingModule::forward shape mismatch".into());
        }
        if let Some(m) = mask
            && m.len() != total
        {
            return Err("RoutingModule::forward mask length mismatch".into());
        }
        if let Some(ref st) = state {
            if mask.is_none() {
                return Err("Mask must be provided if inference state is provided".into());
            }
            if st.batch_size != batch_size || st.d_model != self.d_model {
                return Err("RoutingState dimension mismatch".into());
            }
            if st.has_seen.iter().any(|&seen| seen) {
                return Err("Cannot have seen tokens before prefill forward".into());
            }
        }
        let mut boundary_prob = vec![0.0_f32; total];
        let mut boundary_mask = vec![false; total];
        let mut selected_probs = vec![0.0_f32; total];
        for b in 0..batch_size {
            let base = b * seq_len;
            for t in 0..seq_len {
                let idx = base + t;
                let p = if t == 0 {
                    PAD_PROB
                } else {
                    let prev = &hidden_states[(idx - 1) * self.d_model..idx * self.d_model];
                    let curr = &hidden_states[idx * self.d_model..(idx + 1) * self.d_model];
                    let q = project_norm(&self.q_proj, prev, self.d_model);
                    let k = project_norm(&self.k_proj, curr, self.d_model);
                    cosine_prob(&q, &k)
                };
                let raw_sel = p > 0.5_f32;
                boundary_prob[idx] = p;
                selected_probs[idx] = if raw_sel { p } else { 1.0_f32 - p };
                boundary_mask[idx] = mask.is_none_or(|m| m[idx]) && raw_sel;
            }
            if let (Some(st), Some(m)) = (state.as_deref_mut(), mask) {
                let valid = m[base..base + seq_len].iter().filter(|&&v| v).count();
                if valid > 0 {
                    st.has_seen[b] = true;
                    let last_t = valid - 1;
                    let src_idx = (base + last_t) * self.d_model;
                    let dst_idx = b * self.d_model;
                    st.last_hidden[dst_idx..dst_idx + self.d_model]
                        .copy_from_slice(&hidden_states[src_idx..src_idx + self.d_model]);
                }
            }
        }
        Ok(RoutingOutput {
            boundary_prob,
            boundary_mask,
            selected_probs,
            batch_size,
            seq_len,
        })
    }

    pub fn forward_packed(
        &self,
        hidden_states: &[f32],
        cu_seqlens: &[usize],
    ) -> Result<RoutingOutput, String> {
        if cu_seqlens.len() < 2 || cu_seqlens[0] != 0 {
            return Err("cu_seqlens must start at 0 and contain at least one sequence".into());
        }
        let total = *cu_seqlens.last().unwrap_or(&0);
        if total == 0 || hidden_states.len() != total * self.d_model {
            return Err("RoutingModule::forward_packed shape mismatch".into());
        }
        let mut boundary_prob = vec![0.0_f32; total];
        boundary_prob[0] = PAD_PROB;
        for t in 1..total {
            let prev = &hidden_states[(t - 1) * self.d_model..t * self.d_model];
            let curr = &hidden_states[t * self.d_model..(t + 1) * self.d_model];
            let q = project_norm(&self.q_proj, prev, self.d_model);
            let k = project_norm(&self.k_proj, curr, self.d_model);
            boundary_prob[t] = cosine_prob(&q, &k);
        }
        for &start in &cu_seqlens[..cu_seqlens.len() - 1] {
            if start < total {
                boundary_prob[start] = PAD_PROB;
            }
        }
        let mut boundary_mask = vec![false; total];
        let mut selected_probs = vec![0.0_f32; total];
        for t in 0..total {
            let p = boundary_prob[t];
            let sel = p > 0.5_f32;
            boundary_mask[t] = sel;
            selected_probs[t] = if sel { p } else { 1.0_f32 - p };
        }
        Ok(RoutingOutput {
            boundary_prob,
            boundary_mask,
            selected_probs,
            batch_size: 1,
            seq_len: total,
        })
    }

    pub fn step(
        &self,
        hidden_states: &[f32],
        state: &mut RoutingState,
    ) -> Result<RoutingOutput, String> {
        let batch_size = state.batch_size;
        if hidden_states.len() != batch_size * self.d_model || state.d_model != self.d_model {
            return Err("RoutingModule::step shape mismatch".into());
        }
        let mut boundary_prob = vec![0.0_f32; batch_size];
        let mut boundary_mask = vec![false; batch_size];
        let mut selected_probs = vec![0.0_f32; batch_size];
        for b in 0..batch_size {
            let slice = &hidden_states[b * self.d_model..(b + 1) * self.d_model];
            let prev = &state.last_hidden[b * self.d_model..(b + 1) * self.d_model];
            let p = if state.has_seen[b] {
                let q = project_norm(&self.q_proj, prev, self.d_model);
                let k = project_norm(&self.k_proj, slice, self.d_model);
                cosine_prob(&q, &k)
            } else {
                PAD_PROB
            };
            state.last_hidden[b * self.d_model..(b + 1) * self.d_model].copy_from_slice(slice);
            state.has_seen[b] = true;
            let sel = p > 0.5_f32;
            boundary_prob[b] = p;
            boundary_mask[b] = sel;
            selected_probs[b] = p.max(1.0_f32 - p);
        }
        Ok(RoutingOutput {
            boundary_prob,
            boundary_mask,
            selected_probs,
            batch_size,
            seq_len: 1,
        })
    }

    /// Continuous multi-tile routing forward pass carrying [`RoutingState`]
    /// across arbitrary tile boundaries without resetting boundary probability
    /// at tile starts.
    pub fn forward_continuous(
        &self,
        hidden_states: &[f32],
        batch_size: usize,
        seq_len: usize,
        mask: Option<&[bool]>,
        state: &mut RoutingState,
    ) -> Result<RoutingOutput, String> {
        let total = batch_size * seq_len;
        if batch_size == 0
            || seq_len == 0
            || hidden_states.len() != total * self.d_model
            || state.batch_size != batch_size
            || state.d_model != self.d_model
        {
            return Err("RoutingModule::forward_continuous shape mismatch".into());
        }
        if let Some(m) = mask
            && m.len() != total
        {
            return Err("RoutingModule::forward_continuous mask length mismatch".into());
        }
        let mut boundary_prob = vec![0.0_f32; total];
        let mut boundary_mask = vec![false; total];
        let mut selected_probs = vec![0.0_f32; total];
        for b in 0..batch_size {
            let base = b * seq_len;
            let state_slice = &mut state.last_hidden[b * self.d_model..(b + 1) * self.d_model];
            for t in 0..seq_len {
                let idx = base + t;
                let valid = mask.is_none_or(|m| m[idx]);
                let curr = &hidden_states[idx * self.d_model..(idx + 1) * self.d_model];
                let p = if !state.has_seen[b] {
                    PAD_PROB
                } else {
                    let q = project_norm(&self.q_proj, state_slice, self.d_model);
                    let k = project_norm(&self.k_proj, curr, self.d_model);
                    cosine_prob(&q, &k)
                };
                let raw_sel = p > 0.5_f32;
                boundary_prob[idx] = p;
                selected_probs[idx] = if raw_sel { p } else { 1.0_f32 - p };
                boundary_mask[idx] = valid && raw_sel;
                if valid {
                    state_slice.copy_from_slice(curr);
                    state.has_seen[b] = true;
                }
            }
        }
        Ok(RoutingOutput {
            boundary_prob,
            boundary_mask,
            selected_probs,
            batch_size,
            seq_len,
        })
    }
}

/// Load-balancing ratio loss (`hnet/utils/train.py:load_balancing_loss`).
pub fn ratio_loss(
    output: &RoutingOutput,
    mask: Option<&[bool]>,
    target_ratio: f32,
) -> Result<f32, String> {
    if !target_ratio.is_finite() || target_ratio <= 1.0_f32 {
        return Err("target_ratio N must be finite and > 1.0".into());
    }
    let total = output.boundary_prob.len();
    if total == 0 || output.boundary_mask.len() != total {
        return Err("RoutingOutput cannot be empty".into());
    }
    if let Some(m) = mask
        && m.len() != total
    {
        return Err("mask length mismatch in ratio_loss".into());
    }
    let mut sum_mask = 0.0_f32;
    let mut sum_prob = 0.0_f32;
    let mut count = 0_usize;
    for i in 0..total {
        if mask.is_none_or(|m| m[i]) {
            if output.boundary_mask[i] {
                sum_mask += 1.0_f32;
            }
            sum_prob += output.boundary_prob[i];
            count += 1;
        }
    }
    if count == 0 {
        return Err("No valid tokens in ratio_loss".into());
    }
    let inv_count = 1.0_f32 / (count as f32);
    let true_ratio = sum_mask * inv_count;
    let avg_prob = sum_prob * inv_count;
    let loss = ((1.0_f32 - true_ratio) * (1.0_f32 - avg_prob)
        + true_ratio * avg_prob * (target_ratio - 1.0_f32))
        * (target_ratio / (target_ratio - 1.0_f32));
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_tokenizer() {
        assert_eq!(size_of::<HNetShape>(), 32);
        assert_eq!(align_of::<HNetShape>(), 4);
        assert_eq!(core::mem::offset_of!(HNetShape, vocab_size), 16);
        assert_eq!(core::mem::offset_of!(HNetShape, target_ratio), 24);
        let src = metal_source("");
        assert!(src.contains("struct HNetShape"));
        assert!(src.contains("__builtin_offsetof(HNetShape, target_ratio) == 24"));
        let tok = ByteTokenizer::new();
        assert_eq!(
            tok.encode("hi", true, true),
            vec![BOS_BYTE, b'h', b'i', EOS_BYTE]
        );
        assert_eq!(tok.decode("é 🦀".as_bytes()).unwrap(), "é 🦀");
        assert!(tok.decode(&[0xf0]).is_err());
    }

    #[test]
    fn routing_parity() {
        let router = RoutingModule::new(2);
        let hidden = vec![1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0];
        let mask = vec![true; 4];
        let mut prefill_state = router.allocate_state(1);
        let batch_out = router
            .forward(&hidden, 1, 4, Some(&mask), Some(&mut prefill_state))
            .unwrap();
        assert_eq!(batch_out.boundary_mask, vec![true, false, true, false]);
        let mut step_state = router.allocate_state(1);
        for t in 0..4 {
            let step_out = router
                .step(&hidden[t * 2..(t + 1) * 2], &mut step_state)
                .unwrap();
            assert_eq!(step_out.boundary_mask[0], batch_out.boundary_mask[t]);
            assert!((step_out.boundary_prob[0] - batch_out.boundary_prob[t]).abs() < 1e-6);
            assert!((step_out.selected_probs[0] - batch_out.selected_probs[t]).abs() < 1e-6);
        }
        assert_eq!(step_state, prefill_state);
        let packed_out = router.forward_packed(&hidden, &[0, 4]).unwrap();
        assert_eq!(packed_out.boundary_mask, batch_out.boundary_mask);
    }

    #[test]
    fn dechunk_parity() {
        let router = RoutingModule::new(2);
        let chunk_layer = ChunkLayer::new();
        let dechunk_layer = DeChunkLayer::new(2);
        let ste = SteResidual::new(2);
        let hidden = vec![2.0, 4.0, 2.0, 4.0, -2.0, -4.0, -2.0, -4.0];
        let mask = vec![true; 4];
        let route = router.forward(&hidden, 1, 4, Some(&mask), None).unwrap();
        let chunked = chunk_layer
            .forward(&hidden, &route.boundary_mask, 1, 4, 2)
            .unwrap();
        let mut dechunk_prefill = dechunk_layer.allocate_state(1);
        let dechunked = dechunk_layer
            .forward(
                &chunked.chunks,
                chunked.max_chunks,
                &route.boundary_mask,
                &route.boundary_prob,
                1,
                4,
                Some(&mut dechunk_prefill),
            )
            .unwrap();
        let combined = ste
            .forward(&dechunked, &hidden, &route.selected_probs)
            .unwrap();
        assert_eq!(combined, dechunked);
        let grad_p = ste
            .backward_prob(&vec![1.0_f32; 8], &dechunked, &route.boundary_prob)
            .unwrap();
        assert!(grad_p[0] > 0.0);
        assert!(grad_p[1] < 0.0);
        let loss = ratio_loss(&route, Some(&mask), 2.0).unwrap();
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tiled_parity() {
        let d_model = 4;
        let seq_len = 64;
        let router = RoutingModule::new(d_model);
        let chunk_layer = ChunkLayer::new();
        let dechunk_layer = DeChunkLayer::new(d_model);
        let mut ste = SteResidual::new(d_model);
        for i in 0..d_model * d_model {
            ste.weight[i] = ((i as f32) * 0.13 - 0.5) * 0.25;
        }
        for i in 0..d_model {
            ste.bias[i] = (i as f32) * 0.05 - 0.07;
        }
        // Construct a sequence with repeated runs (so some continuation tiles
        // have zero boundaries and non-boundary starts) and sign flips
        // (boundaries).
        let mut hidden = vec![0.0_f32; seq_len * d_model];
        for t in 0..seq_len {
            let phase = match t {
                0..=9 => 1.0_f32,
                10..=25 => -1.0_f32,
                26..=45 => 1.0_f32,
                _ => -0.5_f32,
            };
            for d in 0..d_model {
                hidden[t * d_model + d] = phase * ((d + 1) as f32) + ((t % 3) as f32) * 0.01;
            }
        }
        let mask = vec![true; seq_len];
        let full_route = router
            .forward(&hidden, 1, seq_len, Some(&mask), None)
            .unwrap();
        let full_chunk = chunk_layer
            .forward(&hidden, &full_route.boundary_mask, 1, seq_len, d_model)
            .unwrap();
        let mut full_dechunk_state = dechunk_layer.allocate_state(1);
        let full_dechunk = dechunk_layer
            .forward(
                &full_chunk.chunks,
                full_chunk.max_chunks,
                &full_route.boundary_mask,
                &full_route.boundary_prob,
                1,
                seq_len,
                Some(&mut full_dechunk_state),
            )
            .unwrap();
        let full_combined = ste
            .forward(&full_dechunk, &hidden, &full_route.selected_probs)
            .unwrap();

        // Partition into irregular tiles including tiles with 0 boundaries.
        let tile_sizes = [5_usize, 4, 7, 6, 11, 8, 9, 14];
        assert_eq!(tile_sizes.iter().sum::<usize>(), seq_len);
        let mut route_state = router.allocate_state(1);
        let mut dechunk_state = dechunk_layer.allocate_state(1);
        let mut tiled_probs = Vec::with_capacity(seq_len);
        let mut tiled_mask = Vec::with_capacity(seq_len);
        let mut tiled_selected = Vec::with_capacity(seq_len);
        let mut tiled_dechunk = Vec::with_capacity(seq_len * d_model);
        let mut tiled_combined = Vec::with_capacity(seq_len * d_model);
        let mut cursor = 0_usize;
        for &t_len in &tile_sizes {
            let h_tile = &hidden[cursor * d_model..(cursor + t_len) * d_model];
            let m_tile = &mask[cursor..cursor + t_len];
            let r_out = router
                .forward_continuous(h_tile, 1, t_len, Some(m_tile), &mut route_state)
                .unwrap();
            let c_out = chunk_layer
                .forward(h_tile, &r_out.boundary_mask, 1, t_len, d_model)
                .unwrap();
            let d_out = dechunk_layer
                .forward_continuous(
                    &c_out.chunks,
                    c_out.max_chunks,
                    &r_out.boundary_mask,
                    &r_out.boundary_prob,
                    1,
                    t_len,
                    &mut dechunk_state,
                )
                .unwrap();
            let comb_out = ste.forward(&d_out, h_tile, &r_out.selected_probs).unwrap();
            tiled_probs.extend_from_slice(&r_out.boundary_prob);
            tiled_mask.extend_from_slice(&r_out.boundary_mask);
            tiled_selected.extend_from_slice(&r_out.selected_probs);
            tiled_dechunk.extend_from_slice(&d_out);
            tiled_combined.extend_from_slice(&comb_out);
            cursor += t_len;
        }
        assert_eq!(tiled_mask, full_route.boundary_mask);
        assert_eq!(tiled_probs, full_route.boundary_prob);
        assert_eq!(tiled_selected, full_route.selected_probs);
        assert_eq!(tiled_dechunk, full_dechunk);
        assert_eq!(tiled_combined, full_combined);
        assert_eq!(dechunk_state, full_dechunk_state);
    }

    #[test]
    fn dechunk_edges() {
        let d_model = 4;
        let chunk_layer = ChunkLayer::new();
        let dechunk_layer = DeChunkLayer::new(d_model);
        // 1. All-boundary input and chunk count > 256 (one partition).
        let seq_len = 320;
        let hidden: Vec<f32> = (0..seq_len * d_model)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.1)
            .collect();
        let all_true = vec![true; seq_len];
        let probs_all = vec![0.85_f32; seq_len];
        let chunked_all = chunk_layer
            .forward(&hidden, &all_true, 1, seq_len, d_model)
            .unwrap();
        assert_eq!(chunked_all.num_chunks, vec![seq_len]);
        let mut full_state = dechunk_layer.allocate_state(1);
        let full_out = dechunk_layer
            .forward(
                &chunked_all.chunks,
                chunked_all.max_chunks,
                &all_true,
                &probs_all,
                1,
                seq_len,
                Some(&mut full_state),
            )
            .unwrap();

        let mut tiled_state = dechunk_layer.allocate_state(1);
        let mut tiled_out = Vec::with_capacity(seq_len * d_model);
        let mut pos = 0;
        for &tile_len in &[1_usize, 3, 17, 259, 40] {
            let h_slice = &hidden[pos * d_model..(pos + tile_len) * d_model];
            let m_slice = &all_true[pos..pos + tile_len];
            let p_slice = &probs_all[pos..pos + tile_len];
            let c_tile = chunk_layer
                .forward(h_slice, m_slice, 1, tile_len, d_model)
                .unwrap();
            let d_tile = dechunk_layer
                .forward_continuous(
                    &c_tile.chunks,
                    c_tile.max_chunks,
                    m_slice,
                    p_slice,
                    1,
                    tile_len,
                    &mut tiled_state,
                )
                .unwrap();
            tiled_out.extend_from_slice(&d_tile);
            pos += tile_len;
        }
        assert_eq!(pos, seq_len);
        assert_eq!(tiled_out, full_out);
        assert_eq!(tiled_state, full_state);

        // 2. Boundary at first byte only, zero-boundary middle tile, and
        //    boundary at final byte only.
        let n2 = 24;
        let h2: Vec<f32> = (0..n2 * d_model).map(|i| (i as f32) * 0.05).collect();
        let mut mask2 = vec![false; n2];
        mask2[0] = true; // first byte boundary
        mask2[n2 - 1] = true; // final byte boundary
        let mut probs2 = vec![0.15_f32; n2];
        probs2[0] = 1.0;
        probs2[n2 - 1] = 0.9;
        let c2 = chunk_layer.forward(&h2, &mask2, 1, n2, d_model).unwrap();
        assert_eq!(c2.num_chunks, vec![2]);
        let mut s2_full = dechunk_layer.allocate_state(1);
        let d2_full = dechunk_layer
            .forward(
                &c2.chunks,
                c2.max_chunks,
                &mask2,
                &probs2,
                1,
                n2,
                Some(&mut s2_full),
            )
            .unwrap();
        let mut s2_tiled = dechunk_layer.allocate_state(1);
        let mut d2_tiled = Vec::new();
        for (start, len) in [(0, 8), (8, 8), (16, 8)] {
            // Tile (8, 8) has 0 boundaries; tile (16, 8) has boundary only at
            // its final byte.
            let ct = chunk_layer
                .forward(
                    &h2[start * d_model..(start + len) * d_model],
                    &mask2[start..start + len],
                    1,
                    len,
                    d_model,
                )
                .unwrap();
            let dt = dechunk_layer
                .forward_continuous(
                    &ct.chunks,
                    ct.max_chunks,
                    &mask2[start..start + len],
                    &probs2[start..start + len],
                    1,
                    len,
                    &mut s2_tiled,
                )
                .unwrap();
            d2_tiled.extend_from_slice(&dt);
        }
        assert_eq!(d2_tiled, d2_full);
        assert_eq!(s2_tiled, s2_full);
    }
}
