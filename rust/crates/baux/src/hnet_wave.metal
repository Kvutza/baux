// Low-level Apple Silicon Metal 4 kernels for H-Net Dynamic Chunking & Byte Readout.
//
// Academic & Open-Source Attribution:
// - Paper: Sukjun Hwang, Brandon Wang, and Albert Gu, "Dynamic Chunking for
//   End-to-End Hierarchical Sequence Modeling", arXiv:2507.07955 (2025).
// - Blog: Cartesia AI, "From Tokens to Bytes, Part 1"
//   (https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1).
// - Upstream Reference: Goomba Lab / Cartesia AI (https://github.com/goombalab/hnet,
//   Apache-2.0 License, hnet/modules/dc.py, hnet/models/hnet.py, hnet/models/mixer_seq.py).

// Fused RoutingModule: projects q(h_{t-1}) and k(h_t), L2-normalizes via SIMD-group
// reductions, computes p_t = clamp((1 - cos_sim) / 2, 0, 1) with p_0 = 1.0, and
// outputs boundary_prob, boundary_mask, and selected_probs in one pass.
kernel void hnet_route(
    device const float* hidden [[buffer(0)]],
    device const float* q_proj [[buffer(1)]],
    device const float* k_proj [[buffer(2)]],
    device const uint* mask [[buffer(3)]],
    device float* boundary_prob [[buffer(4)]],
    device uint* boundary_mask [[buffer(5)]],
    device float* selected_probs [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint total = shape.batch_size * shape.seq_len;
    if (row >= total) return;
    const uint t = row % shape.seq_len;
    const uint D = shape.d_model;

    float p = 1.0f;
    if (t > 0u) {
        device const float* prev = hidden + ulong(row - 1u) * D;
        device const float* curr = hidden + ulong(row) * D;
        float local_qq = 0.0f;
        float local_kk = 0.0f;
        float local_qk = 0.0f;
        for (uint r = lane; r < D; r += 32u) {
            device const float* wq = q_proj + ulong(r) * D;
            device const float* wk = k_proj + ulong(r) * D;
            float q_r = 0.0f;
            float k_r = 0.0f;
            for (uint c = 0u; c < D; ++c) {
                q_r = fma(wq[c], prev[c], q_r);
                k_r = fma(wk[c], curr[c], k_r);
            }
            local_qq = fma(q_r, q_r, local_qq);
            local_kk = fma(k_r, k_r, local_kk);
            local_qk = fma(q_r, k_r, local_qk);
        }
        const float norm_q = max(sqrt(simd_sum(local_qq)), 1e-12f);
        const float norm_k = max(sqrt(simd_sum(local_kk)), 1e-12f);
        const float dot_qk = simd_sum(local_qk);
        const float cos_sim = dot_qk / (norm_q * norm_k);
        p = clamp((1.0f - cos_sim) * 0.5f, 0.0f, 1.0f);
    }

    if (lane == 0u) {
        const bool raw_sel = p > 0.5f;
        boundary_prob[row] = p;
        selected_probs[row] = raw_sel ? p : (1.0f - p);
        boundary_mask[row] = (mask[row] != 0u && raw_sel) ? 1u : 0u;
    }
}

// Single-pass SIMD prefix-sum compaction (ChunkLayer) + plug_back_idx generation.
// Replaces upstream's O(L log L) torch.argsort with hardware simd_prefix_exclusive_sum.
kernel void hnet_chunk_scan(
    device const float* hidden [[buffer(0)]],
    device const uint* boundary_mask [[buffer(1)]],
    device float* chunks [[buffer(2)]],
    device uint* plug_idx [[buffer(3)]],
    device uint* num_chunks [[buffer(4)]],
    constant HNetShape& shape [[buffer(5)]],
    uint b [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (b >= shape.batch_size) return;
    const uint L = shape.seq_len;
    const uint D = shape.d_model;
    const uint M = shape.max_chunks;

    device const float* b_hidden = hidden + ulong(b) * L * D;
    device const uint* b_mask = boundary_mask + ulong(b) * L;
    device float* b_chunks = chunks + ulong(b) * M * D;
    device uint* b_plug = plug_idx + ulong(b) * L;

    uint running = 0u;
    for (uint base = 0u; base < L; base += 32u) {
        const uint t = base + lane;
        const uint is_b = (t < L && b_mask[t] != 0u) ? 1u : 0u;
        const uint prefix = simd_prefix_exclusive_sum(is_b);
        const uint write_idx = running + prefix;
        if (t < L) {
            const uint inclusive = write_idx + is_b;
            b_plug[t] = inclusive > 0u ? (inclusive - 1u) : 0u;
            if (is_b != 0u && write_idx < M) {
                device const float* src = b_hidden + ulong(t) * D;
                device float* dst = b_chunks + ulong(write_idx) * D;
                for (uint d = 0u; d < D; ++d) {
                    dst[d] = src[d];
                }
            }
        }
        running += simd_sum(is_b);
    }

    uint pad_cursor = running;
    for (uint base = 0u; base < L && pad_cursor < M; base += 32u) {
        const uint t = base + lane;
        const uint is_pad = (t < L && b_mask[t] == 0u) ? 1u : 0u;
        const uint prefix = simd_prefix_exclusive_sum(is_pad);
        const uint write_idx = pad_cursor + prefix;
        if (t < L && is_pad != 0u && write_idx < M) {
            device const float* src = b_hidden + ulong(t) * D;
            device float* dst = b_chunks + ulong(write_idx) * D;
            for (uint d = 0u; d < D; ++d) {
                dst[d] = src[d];
            }
        }
        pad_cursor += simd_sum(is_pad);
    }

    if (lane == 0u) {
        num_chunks[b] = running;
    }
}

// Fused register EMA scan + plug-back gather + STE residual projection (DeChunkLayer + STE).
// Each thread owns one (b, d) channel, maintaining the EMA state in a register across t=0..L-1.
kernel void hnet_dechunk_ste(
    device const float* chunks [[buffer(0)]],
    device const uint* boundary_mask [[buffer(1)]],
    device const float* boundary_prob [[buffer(2)]],
    device const float* encoder_out [[buffer(3)]],
    device const float* res_weight [[buffer(4)]],
    device const float* res_bias [[buffer(5)]],
    device const float* selected_probs [[buffer(6)]],
    device float* dechunk_out [[buffer(7)]],
    device float* combined_out [[buffer(8)]],
    constant HNetShape& shape [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    const uint D = shape.d_model;
    const uint total_channels = shape.batch_size * D;
    if (gid >= total_channels) return;

    const uint b = gid / D;
    const uint d = gid % D;
    const uint L = shape.seq_len;
    const uint M = shape.max_chunks;

    device const float* b_chunks = chunks + ulong(b) * M * D;
    device const uint* b_mask = boundary_mask + ulong(b) * L;
    device const float* b_prob = boundary_prob + ulong(b) * L;
    device const float* b_enc = encoder_out + ulong(b) * L * D;
    device const float* b_sel = selected_probs + ulong(b) * L;
    device const float* w_row = res_weight + ulong(d) * D;
    const float bias_d = res_bias[d];

    device float* b_dechunk = dechunk_out + ulong(b) * L * D;
    device float* b_combined = combined_out + ulong(b) * L * D;

    float ema_state = 0.0f;
    uint chunk_cursor = 0u;
    for (uint t = 0u; t < L; ++t) {
        if (b_mask[t] != 0u && chunk_cursor < M) {
            const float p = clamp(b_prob[t], shape.eps, 1.0f - shape.eps);
            const float z_val = b_chunks[ulong(chunk_cursor) * D + d];
            ema_state = fma(p, z_val, (1.0f - p) * ema_state);
            chunk_cursor += 1u;
        }
        b_dechunk[ulong(t) * D + d] = ema_state;

        device const float* enc_t = b_enc + ulong(t) * D;
        float res = bias_d;
        for (uint k = 0u; k < D; ++k) {
            res = fma(w_row[k], enc_t[k], res);
        }
        const float ste_one = 1.0f + 0.0f * b_sel[t];
        b_combined[ulong(t) * D + d] = fma(ema_state, ste_one, res);
    }
}

// Exact 256-way byte-level readout + SIMD-group online log-sum-exp cross-entropy loss.
// Replaces the 64-dim 6-level base-8 centroid tree approximation (mhc4_readout) with
// an exact full-d_model 256-byte softmax in 8 registers per lane of a 32-lane SIMD group.
kernel void hnet_byte_readout(
    device const float* hidden [[buffer(0)]],
    device const float* lm_head [[buffer(1)]],
    device const uint* targets [[buffer(2)]],
    device uint* predictions [[buffer(3)]],
    device float* losses [[buffer(4)]],
    constant HNetShape& shape [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint total = shape.batch_size * shape.seq_len;
    if (row >= total) return;
    const uint D = shape.d_model;
    device const float* h_row = hidden + ulong(row) * D;

    float logits[8];
    float local_max = -INFINITY;
    for (uint k = 0u; k < 8u; ++k) {
        const uint vocab_idx = lane + k * 32u;
        device const float* w_row = lm_head + ulong(vocab_idx) * D;
        float acc = 0.0f;
        for (uint d = 0u; d < D; ++d) {
            acc = fma(w_row[d], h_row[d], acc);
        }
        logits[k] = acc;
        local_max = max(local_max, acc);
    }

    const float max_logit = simd_max(local_max);
    float local_sum_exp = 0.0f;
    uint local_best_vocab = 256u;
    for (uint k = 0u; k < 8u; ++k) {
        const uint vocab_idx = lane + k * 32u;
        local_sum_exp += exp(logits[k] - max_logit);
        if (logits[k] == max_logit && vocab_idx < local_best_vocab) {
            local_best_vocab = vocab_idx;
        }
    }
    const float sum_exp = simd_sum(local_sum_exp);
    const uint best_vocab = simd_min(local_best_vocab);

    const uint target = targets[row] & 255u;
    const uint target_lane = target & 31u;
    const uint target_k = target >> 5u;
    const float lane_target_logit = (lane == target_lane) ? logits[target_k] : 0.0f;
    const float target_logit = simd_sum(lane_target_logit);

    if (lane == 0u) {
        predictions[row] = best_vocab;
        losses[row] = (shape.score != 0u) ? ((max_logit + log(sum_exp)) - target_logit) : 0.0f;
    }
}

// Fused Straight-Through Estimator (STE) backward pass for router probability P_t.
// Computes grad_p[row] = (boundary_mask[row] != 0 ? +1 : -1) * dot(grad_combined[row], dechunk_out[row])
// using a 32-lane SIMD-group reduction.
kernel void hnet_ste_bwd(
    device const float* grad_combined [[buffer(0)]],
    device const float* dechunk_out [[buffer(1)]],
    device const uint* boundary_mask [[buffer(2)]],
    device float* grad_prob [[buffer(3)]],
    constant HNetShape& shape [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint total = shape.batch_size * shape.seq_len;
    if (row >= total) return;
    const uint D = shape.d_model;
    device const float* g_row = grad_combined + ulong(row) * D;
    device const float* y_row = dechunk_out + ulong(row) * D;

    float local_dot = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        local_dot = fma(g_row[d], y_row[d], local_dot);
    }
    const float dot_val = simd_sum(local_dot);
    if (lane == 0u) {
        grad_prob[row] = (boundary_mask[row] != 0u) ? dot_val : -dot_val;
    }
}

// Fused FP16 4-stream mHC-4 dynamic chunking router and prefix-sum compaction.
// Computes Cartesia H-Net cosine-similarity boundary probabilities on Layer-0
// normalized activations [L, 512], saves the full-resolution 4-stream state
// [L, 2048], and compacts selected boundary rows into [chunk_rows, 2048].
kernel void hnet_mhc_chunk(
    device const half* normalized [[buffer(0)]],
    device half* streams [[buffer(1)]],
    device half* encoder_streams [[buffer(2)]],
    device uint* origins [[buffer(3)]],
    device float* probs [[buffer(4)]],
    device uint* plug_idx [[buffer(5)]],
    constant uint2& counts [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint L = counts.x;
    const uint chunk_rows = counts.y;
    if (L == 0u || chunk_rows == 0u || L > 4096u) return;

    threadgroup float shared_prob[4096];
    threadgroup ushort shared_origin[4096];
    threadgroup float shared_tau;

    const uint total_vec4 = L * 512u;
    device const half4* src_streams4 = reinterpret_cast<device const half4*>(streams);
    device half4* enc_streams4 = reinterpret_cast<device half4*>(encoder_streams);
    for (uint idx = tid; idx < total_vec4; idx += 256u) {
        enc_streams4[idx] = src_streams4[idx];
    }

    for (uint t = sg; t < L; t += 8u) {
        if (t == 0u) {
            if (lane == 0u) shared_prob[0] = 1.0f;
        } else {
            device const half4* prev4 =
                reinterpret_cast<device const half4*>(normalized + ulong(t - 1u) * 512u);
            device const half4* curr4 =
                reinterpret_cast<device const half4*>(normalized + ulong(t) * 512u);
            float dot_acc = 0.0f;
            float norm_prev = 0.0f;
            float norm_curr = 0.0f;
            for (uint k = lane; k < 128u; k += 32u) {
                const float4 a = float4(prev4[k]);
                const float4 b = float4(curr4[k]);
                dot_acc = fma(a.x, b.x, fma(a.y, b.y, fma(a.z, b.z, fma(a.w, b.w, dot_acc))));
                norm_prev = fma(a.x, a.x, fma(a.y, a.y, fma(a.z, a.z, fma(a.w, a.w, norm_prev))));
                norm_curr = fma(b.x, b.x, fma(b.y, b.y, fma(b.z, b.z, fma(b.w, b.w, norm_curr))));
            }
            const float qk = simd_sum(dot_acc);
            const float nq = simd_sum(norm_prev);
            const float nk = simd_sum(norm_curr);
            if (lane == 0u) {
                const float denom = max(sqrt(nq) * sqrt(nk), 1.0e-8f);
                const float cos_sim = clamp(qk / denom, -1.0f, 1.0f);
                shared_prob[t] = clamp((1.0f - cos_sim) * 0.5f, 0.0f, 1.0f);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (sg == 0u) {
        float tau = 0.5f;
        uint local_c = 0u;
        for (uint t = 1u + lane; t < L; t += 32u) {
            local_c += uint(shared_prob[t] >= tau);
        }
        if (1u + simd_sum(local_c) > chunk_rows) {
            float lo = 0.5f;
            float hi = 1.0f;
            for (uint iter = 0u; iter < 8u; ++iter) {
                const float mid = 0.5f * (lo + hi);
                uint c = 0u;
                for (uint t = 1u + lane; t < L; t += 32u) {
                    c += uint(shared_prob[t] >= mid);
                }
                if (1u + simd_sum(c) > chunk_rows) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            tau = hi;
        }
        if (lane == 0u) shared_tau = tau;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        const float tau = shared_tau;
        uint count = 0u;
        for (uint t = 0u; t < L; ++t) {
            const float p = shared_prob[t];
            const bool open = (t == 0u) || (p >= tau && count < chunk_rows);
            if (open) {
                shared_origin[count] = ushort(t);
                origins[count] = t;
                probs[count] = p;
                count += 1u;
            }
            plug_idx[t] = count - 1u;
        }
        const ushort last_origin = shared_origin[count - 1u];
        for (uint c = count; c < chunk_rows; ++c) {
            shared_origin[c] = last_origin;
            origins[c] = uint(last_origin);
            probs[c] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* dst_streams4 = reinterpret_cast<device half4*>(streams);
    const uint compact_vec4 = chunk_rows * 512u;
    for (uint idx = tid; idx < compact_vec4; idx += 256u) {
        const uint row = idx >> 9;
        const uint col4 = idx & 511u;
        const uint src_row = uint(shared_origin[row]);
        dst_streams4[idx] = enc_streams4[src_row * 512u + col4];
    }
}

// Fused FP16 4-stream H-Net EMA DeChunk scan + plug-back + STE residual combine.
// Each thread owns one half4 (4 adjacent FP16 channels in 0..2048), running the
// register-resident EMA scan over [0..chunk_rows) and plugging back into [0..L).
kernel void hnet_mhc_dechunk(
    device half* streams [[buffer(0)]],
    device const half* encoder_streams [[buffer(1)]],
    device half* chunk_scratch [[buffer(2)]],
    device const uint* origins [[buffer(3)]],
    device const float* probs [[buffer(4)]],
    device const uint* plug_idx [[buffer(5)]],
    constant uint2& counts [[buffer(6)]],
    uint col4 [[thread_position_in_grid]]) {
    const uint L = counts.x;
    const uint chunk_rows = counts.y;
    if (col4 >= 512u || L == 0u || chunk_rows == 0u) return;

    device half4* streams4 = reinterpret_cast<device half4*>(streams);
    device const half4* encoder4 = reinterpret_cast<device const half4*>(encoder_streams);
    device half4* scratch4 = reinterpret_cast<device half4*>(chunk_scratch);

    float4 ema_h = float4(0.0f);
    for (uint i = 0u; i < chunk_rows; ++i) {
        const float raw_p = probs[i];
        if (raw_p > 0.0f) {
            const uint t_i = origins[i];
            const float4 z_hat = float4(streams4[ulong(i) * 512u + col4]);
            const float4 z_enc = float4(encoder4[ulong(t_i) * 512u + col4]);
            const float4 delta = z_hat - z_enc;
            const float p = clamp(raw_p, 1.0e-4f, 1.0f - 1.0e-4f);
            ema_h = fma(float4(p), delta, float4(1.0f - p) * ema_h);
        }
        scratch4[ulong(i) * 512u + col4] = half4(ema_h);
    }

    for (uint t = 0u; t < L; ++t) {
        const uint i = plug_idx[t];
        const float4 ema_delta = float4(scratch4[ulong(i) * 512u + col4]);
        const float4 z_enc = float4(encoder4[ulong(t) * 512u + col4]);
        streams4[ulong(t) * 512u + col4] = half4(z_enc + ema_delta);
    }
}
