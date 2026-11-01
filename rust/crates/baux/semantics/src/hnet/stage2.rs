//! Sequential 512-dimensional 2-stage H-Net reference oracle (`E0 -> Route0 ->
//! Chunk0 -> E1 -> Route1 -> Chunk1 -> Core -> Dechunk1 -> D1 -> Dechunk0 ->
//! STE0 -> D0 -> Readout`) and combined 2-stage Cartesia ratio loss.

use super::{BYTE_VOCAB, EMA_EPS, NORM_EPS, PAD_PROB};

/// Parameters for the 2-stage learned causal H-Net reference oracle.
pub struct HNet2Params<'a> {
    pub d_model: usize,
    pub u0_table: &'a [f32],
    pub gates_table: &'a [f32],
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_res: &'a [f32],
    pub b_res: &'a [f32],
    pub attn_norm1: &'a [f32],
    pub ffn_norm1: &'a [f32],
    pub attn_norm3: &'a [f32],
    pub ffn_norm3: &'a [f32],
    pub ffn_norm4: &'a [f32],
    pub final_norm: &'a [f32],
    pub readout: &'a [f32],
    pub gamma1: f32,
    pub gamma3: f32,
}

/// Forward hierarchy outputs of the 2-stage learned causal H-Net.
#[derive(Clone, Debug, PartialEq)]
pub struct HNet2Forward {
    pub h0: Vec<f32>,
    pub probs0: Vec<f32>,
    pub origins0: Vec<u32>,
    pub plug0: Vec<u32>,
    pub z0: Vec<f32>,
    pub h1: Vec<f32>,
    pub probs1: Vec<f32>,
    pub origins1: Vec<u32>,
    pub plug1: Vec<u32>,
    pub z1: Vec<f32>,
}

/// Reverse hierarchy and readout outputs of the 2-stage learned causal H-Net.
#[derive(Clone, Debug, PartialEq)]
pub struct HNet2Reverse {
    pub ema1: Vec<f32>,
    pub y1: Vec<f32>,
    pub ema0: Vec<f32>,
    pub y0: Vec<f32>,
    pub predictions: Vec<u32>,
    pub losses: Vec<f32>,
    pub ratio_loss: f32,
    pub ratio_penalty: f32,
}

fn mat_vec(w: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    for (row, dst) in out.iter_mut().enumerate() {
        let slice = &w[row * d..(row + 1) * d];
        let mut acc = 0.0_f32;
        for (&wi, &xi) in slice.iter().zip(x.iter()) {
            acc = wi.mul_add(xi, acc);
        }
        *dst = acc;
    }
    out
}

fn stage0_proj(w: &[f32], x: &[f32], c_id: f32, shift: usize) -> Vec<f32> {
    let d = x.len();
    let mut out = vec![0.0_f32; d];
    for (row, dst) in out.iter_mut().enumerate() {
        let slice = &w[row * d..(row + 1) * d];
        let mut acc = 0.0_f32;
        for (&wi, &xi) in slice.iter().zip(x.iter()) {
            acc = wi.mul_add(xi, acc);
        }
        let base = 0.35_f32 * x[row] + x[(row + shift) % d];
        *dst = 1.00_f32 * acc + c_id * (base - 1.00_f32 * x[row]);
    }
    out
}

fn stage1_proj(w: &[f32], norm: &[f32], x: &[f32], c_id: f32, shift: usize) -> Vec<f32> {
    let d = x.len();
    let mut out = vec![0.0_f32; d];
    for (row, dst) in out.iter_mut().enumerate() {
        let slice = &w[row * d..(row + 1) * d];
        let mut acc = 0.0_f32;
        for (&wi, &xi) in slice.iter().zip(x.iter()) {
            acc = wi.mul_add(xi, acc);
        }
        let shifted = 0.15_f32 * x[row] + x[(row + shift) % d];
        let val = 0.35_f32 * acc + c_id * (shifted - 0.35_f32 * x[row]);
        *dst = norm[row] * val;
    }
    out
}

fn cos_prob(q: &[f32], k: &[f32]) -> f32 {
    let (mut dot, mut nq, mut nk) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (&qi, &ki) in q.iter().zip(k.iter()) {
        dot = qi.mul_add(ki, dot);
        nq = qi.mul_add(qi, nq);
        nk = ki.mul_add(ki, nk);
    }
    let denom = nq.sqrt().max(NORM_EPS) * nk.sqrt().max(NORM_EPS);
    ((1.0_f32 - dot / denom) * 0.5_f32).clamp(0.0_f32, 1.0_f32)
}

fn run_stage0(
    params: &HNet2Params<'_>,
    tokens: &[u32],
    c_id: f32,
) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
    let d = params.d_model;
    let n = tokens.len();
    let mut h0 = vec![0.0_f32; n * d];
    let mut probs0 = vec![0.0_f32; n];
    let mut origins0 = Vec::new();
    let mut plug0 = vec![0_u32; n];
    let mut z0 = Vec::new();
    let mut state = vec![0.0_f32; d];
    for (t, &tok) in tokens.iter().enumerate() {
        let b = (tok as usize) % BYTE_VOCAB;
        let alpha = params.gates_table[b * 2];
        let one_minus = 1.0_f32 - alpha;
        let u0 = &params.u0_table[b * d..(b + 1) * d];
        for col in 0..d {
            state[col] = alpha.mul_add(state[col], one_minus * u0[col]);
        }
        h0[t * d..(t + 1) * d].copy_from_slice(&state);
        let p = if t == 0 {
            PAD_PROB
        } else {
            let q = stage0_proj(params.w_q, &h0[(t - 1) * d..t * d], c_id, 3);
            let k = stage0_proj(params.w_k, &state, c_id, 4);
            cos_prob(&q, &k)
        };
        probs0[t] = p;
        if t == 0 || p > 0.5_f32 {
            origins0.push(t as u32);
            z0.extend_from_slice(&state);
        }
        plug0[t] = (origins0.len() - 1) as u32;
    }
    (h0, probs0, origins0, plug0, z0)
}

/// Run the sequential 512-dimensional 2-stage H-Net forward hierarchy.
pub fn hnet2_forward(params: &HNet2Params<'_>, tokens: &[u32]) -> Result<HNet2Forward, String> {
    if tokens.is_empty() || params.d_model == 0 {
        return Err("hnet2_forward requires non-empty tokens and positive d_model".into());
    }
    let d = params.d_model;
    let trace_q: f32 = (0..d).map(|i| params.w_q[i * d + i]).sum();
    let c_id = if trace_q > (d as f32) * 0.5_f32 {
        1.0_f32
    } else {
        0.0_f32
    };
    let (h0, probs0, origins0, plug0, z0) = run_stage0(params, tokens, c_id);
    let m0 = origins0.len();
    let mut h1 = vec![0.0_f32; m0 * d];
    let mut probs1 = vec![0.0_f32; m0];
    let mut origins1 = Vec::new();
    let mut plug1 = vec![0_u32; m0];
    let mut z1 = Vec::new();
    let mut state1 = vec![0.0_f32; d];
    let g1 = params.gamma1;
    let one_minus_g1 = 1.0_f32 - g1;
    for c0 in 0..m0 {
        let z0_slice = &z0[c0 * d..(c0 + 1) * d];
        for col in 0..d {
            let u1 = params.attn_norm1[col] * z0_slice[col];
            state1[col] = g1.mul_add(state1[col], one_minus_g1 * u1);
        }
        h1[c0 * d..(c0 + 1) * d].copy_from_slice(&state1);
        let p1 = if c0 == 0 {
            PAD_PROB
        } else {
            let prev = &h1[(c0 - 1) * d..c0 * d];
            let q1 = stage1_proj(params.w_q, params.ffn_norm1, prev, c_id, 1);
            let k1 = stage1_proj(params.w_k, params.ffn_norm1, &state1, c_id, 5);
            cos_prob(&q1, &k1)
        };
        probs1[c0] = p1;
        if c0 == 0 || p1 > 0.5_f32 {
            origins1.push(c0 as u32);
            z1.extend_from_slice(&state1);
        }
        plug1[c0] = (origins1.len() - 1) as u32;
    }
    Ok(HNet2Forward {
        h0,
        probs0,
        origins0,
        plug0,
        z0,
        h1,
        probs1,
        origins1,
        plug1,
        z1,
    })
}

/// Compute the combined Stage-0 ($N_0 = 4.0$) and Stage-1 ($N_1 = 2.0$)
/// Cartesia load-balancing ratio loss and penalty.
pub fn hnet2_penalty(probs0: &[f32], m0: usize, probs1: &[f32], m1: usize) -> (f32, f32) {
    let n0 = (probs0.len().max(1)) as f32;
    let f0 = (m0 as f32) / n0;
    let g0 = (probs0.iter().map(|&p| f64::from(p)).sum::<f64>() as f32) / n0;
    let t0 = 4.0_f32;
    let rl0 = |f: f32, g: f32| (t0 / (t0 - 1.0)) * ((1.0 - f) * (1.0 - g) + (t0 - 1.0) * f * g);
    let loss0 = rl0(f0, g0);
    let dev0 = 0.5_f32 * ((rl0(f0, f0) - 1.0).max(0.0) + (rl0(g0, g0) - 1.0).max(0.0));

    let denom1 = (probs1.len().max(1)) as f32;
    let f1 = (m1 as f32) / denom1;
    let g1 = (probs1.iter().map(|&p| f64::from(p)).sum::<f64>() as f32) / denom1;
    let t1 = 2.0_f32;
    let rl1 = |f: f32, g: f32| (t1 / (t1 - 1.0)) * ((1.0 - f) * (1.0 - g) + (t1 - 1.0) * f * g);
    let loss1 = rl1(f1, g1);
    let dev1 = 0.5_f32 * ((rl1(f1, f1) - 1.0).max(0.0) + (rl1(g1, g1) - 1.0).max(0.0));
    (loss0 + loss1, 0.03_f32 * (dev0 + dev1))
}

fn dechunk_stages(
    params: &HNet2Params<'_>,
    fwd: &HNet2Forward,
    core: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = params.d_model;
    let m0 = fwd.origins0.len();
    let m1 = fwd.origins1.len();
    let mut ema1 = vec![0.0_f32; m1 * d];
    let mut state_e1 = vec![0.0_f32; d];
    for c1 in 0..m1 {
        let p = fwd.probs1[fwd.origins1[c1] as usize].clamp(EMA_EPS, 1.0_f32 - EMA_EPS);
        let one_minus = 1.0_f32 - p;
        for col in 0..d {
            state_e1[col] = p.mul_add(core[c1 * d + col], one_minus * state_e1[col]);
            ema1[c1 * d + col] = state_e1[col];
        }
    }
    let mut y1 = vec![0.0_f32; m0 * d];
    let mut ema0 = vec![0.0_f32; m0 * d];
    let mut state_y1 = vec![0.0_f32; d];
    let mut state_e0 = vec![0.0_f32; d];
    let g3 = params.gamma3;
    let one_minus_g3 = 1.0_f32 - g3;
    for c0 in 0..m0 {
        let c1_idx = fwd.plug1[c0] as usize;
        let p0 = fwd.probs0[fwd.origins0[c0] as usize].clamp(EMA_EPS, 1.0_f32 - EMA_EPS);
        let one_minus_p0 = 1.0_f32 - p0;
        for col in 0..d {
            let e1_val = ema1[c1_idx * d + col];
            let h1_val = fwd.h1[c0 * d + col];
            let z1_val = params.ffn_norm3[col] * (e1_val + params.attn_norm3[col] * h1_val);
            state_y1[col] = g3.mul_add(state_y1[col], one_minus_g3 * z1_val);
            y1[c0 * d + col] = state_y1[col];
            state_e0[col] = p0.mul_add(state_y1[col], one_minus_p0 * state_e0[col]);
            ema0[c0 * d + col] = state_e0[col];
        }
    }
    (ema1, y1, ema0)
}

/// Run the sequential 512-dimensional 2-stage H-Net reverse hierarchy and
/// 256-way byte readout.
pub fn hnet2_reverse(
    params: &HNet2Params<'_>,
    fwd: &HNet2Forward,
    core: &[f32],
    tokens: &[u32],
    labels: &[u32],
) -> Result<HNet2Reverse, String> {
    let d = params.d_model;
    let n = tokens.len();
    let m1 = fwd.origins1.len();
    if labels.len() != n || core.len() != m1 * d {
        return Err("hnet2_reverse shape mismatch".into());
    }
    let (ema1, y1, ema0) = dechunk_stages(params, fwd, core);
    let mut y0 = vec![0.0_f32; n * d];
    let mut predictions = vec![0_u32; n];
    let mut losses = vec![0.0_f32; n];
    let mut state_y0 = vec![0.0_f32; d];
    for t in 0..n {
        let b = (tokens[t] as usize) % BYTE_VOCAB;
        let beta = params.gates_table[b * 2 + 1];
        let one_minus = 1.0_f32 - beta;
        let r0 = mat_vec(params.w_res, &fwd.h0[t * d..(t + 1) * d], d);
        let c0_idx = fwd.plug0[t] as usize;
        for col in 0..d {
            let z0_val = ema0[c0_idx * d + col] + r0[col] + params.b_res[col];
            state_y0[col] = beta.mul_add(state_y0[col], one_minus * z0_val);
            y0[t * d + col] = state_y0[col];
        }
        let (pred, loss) = byte_loss(params, &state_y0, labels[t]);
        predictions[t] = pred;
        losses[t] = loss;
    }
    let (ratio_loss, ratio_penalty) =
        hnet2_penalty(&fwd.probs0, fwd.origins0.len(), &fwd.probs1, m1);
    Ok(HNet2Reverse {
        ema1,
        y1,
        ema0,
        y0,
        predictions,
        losses,
        ratio_loss,
        ratio_penalty,
    })
}

fn byte_loss(params: &HNet2Params<'_>, y0: &[f32], label: u32) -> (u32, f32) {
    let d = params.d_model;
    let mut logits = [0.0_f32; BYTE_VOCAB];
    let mut top = f32::NEG_INFINITY;
    let mut best = 0_u32;
    for (v, slot) in logits.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for col in 0..d {
            let scaled = params.final_norm[col] * params.ffn_norm4[col] * y0[col];
            sum = scaled.mul_add(params.readout[col * BYTE_VOCAB + v], sum);
        }
        *slot = sum;
        if sum > top {
            top = sum;
            best = v as u32;
        }
    }
    let total: f32 = logits.iter().map(|&l| (l - top).exp()).sum();
    let target = logits[(label as usize) % BYTE_VOCAB];
    (best, top + total.ln() - target)
}
