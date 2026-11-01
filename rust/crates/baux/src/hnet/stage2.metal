// Two-stage 512-dimensional learned causal H-Net kernels on Apple Silicon Metal.
#include <metal_simdgroup_matrix>

inline float hnet2_sig(float x) {
    const float e = exp(-abs(x));
    return max(x >= 0.0f ? (1.0f / (1.0f + e)) : (e / (1.0f + e)), 1e-20f);
}

kernel void hnet2_prep_mean(
    device const half* readout [[buffer(0)]],
    device const half* qkv0 [[buffer(1)]],
    device const half* output0 [[buffer(2)]],
    device const half* hnet_w [[buffer(3)]],
    device const half* attn_ctrl [[buffer(4)]],
    device const half* moe_ctrl [[buffer(5)]],
    device float* mean_z [[buffer(6)]],
    device float* scalars [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scale_b[256];
    threadgroup float mean_e[512];
    threadgroup float mean_v[64];

    if (tid < 256u) {
        float sq = 0.0f;
        for (uint d = 0u; d < 512u; ++d) {
            const float v = float(readout[d * 256u + tid]);
            sq = fma(v, v, sq);
        }
        scale_b[tid] = rsqrt(sq / 512.0f + 1e-5f) / 256.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < 512u; d += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < 256u; ++b) {
            acc = fma(float(readout[d * 256u + b]), scale_b[b], acc);
        }
        mean_e[d] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        float acc = 0.0f;
        for (uint d = 0u; d < 512u; ++d) {
            acc = fma(mean_e[d], float(qkv0[d * 640u + 576u + tid]), acc);
        }
        mean_v[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < 512u; d += 256u) {
        float mz = 0.0f;
        for (uint h = 0u; h < 8u; ++h) {
            for (uint c = 0u; c < 64u; ++c) {
                const uint row_off = (64u * h + c) * 512u;
                mz = fma(mean_v[c], float(output0[row_off + d]), mz);
            }
        }
        mean_z[d] = mz;
    }

    if (tid == 0u) {
        float c1 = 0.0f, c3 = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            c1 += float(attn_ctrl[4u + j]) + float(moe_ctrl[4u + j]);
            c3 += float(attn_ctrl[12u + j]) + float(moe_ctrl[12u + j]);
        }
        const float gamma1 = 0.04f + 0.06f * hnet2_sig(0.25f * c1 - 1.0f);
        const float gamma3 = 0.08f + 0.16f * hnet2_sig(0.25f * c3 - 1.0f);
        float trace_q = 0.0f;
        for (uint d = 0u; d < 512u; ++d) {
            trace_q += float(hnet_w[d * 512u + d]);
        }
        scalars[0] = gamma1;
        scalars[1] = gamma3;
        scalars[2] = trace_q > 256.0f ? 1.0f : 0.0f;
        scalars[3] = 0.0f;
    }
}

kernel void hnet2_prep_byte(
    device const half* readout [[buffer(0)]],
    device const float* mean_z [[buffer(1)]],
    device const half* router0 [[buffer(2)]],
    device const half* gate_up0 [[buffer(3)]],
    device const half* down0 [[buffer(4)]],
    device const half* attn_norm [[buffer(5)]],
    device const half* ffn_norm [[buffer(6)]],
    device const half* attn_ctrl [[buffer(7)]],
    device const half* moe_ctrl [[buffer(8)]],
    device float* u0_table [[buffer(9)]],
    device float* gates_table [[buffer(10)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (b >= 256u) return;
    threadgroup float state[512];
    threadgroup float norm_mid[512];
    threadgroup float r_scores[64];
    threadgroup float sg_sq[8];
    threadgroup uint active_exp[4];
    threadgroup float active_w[4];
    threadgroup float act[256];

    float local_sq = 0.0f;
    for (uint d = tid; d < 512u; d += 256u) {
        const float v = float(readout[d * 256u + b]) + mean_z[d];
        state[d] = v;
        local_sq = fma(v, v, local_sq);
    }
    local_sq = simd_sum(local_sq);
    if (lane == 0u) sg_sq[sg_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    for (uint s = 0u; s < 8u; ++s) total_sq += sg_sq[s];
    const float rms_scale = rsqrt(total_sq / 512.0f + 1e-5f);
    for (uint d = tid; d < 512u; d += 256u) {
        norm_mid[d] = state[d] * rms_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = tid; r < 64u; r += 256u) {
        float s = 0.0f;
        for (uint d = 0u; d < 512u; ++d) {
            s = fma(norm_mid[d], float(router0[d * 64u + r]), s);
        }
        r_scores[r] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float sc0 = -INFINITY, sc1 = -INFINITY, sc2 = -INFINITY;
        uint ix0 = 0u, ix1 = 0u, ix2 = 0u;
        for (uint r = 0u; r < 64u; ++r) {
            const float s = r_scores[r];
            if (s > sc0) {
                sc2 = sc1; ix2 = ix1;
                sc1 = sc0; ix1 = ix0;
                sc0 = s; ix0 = r;
            } else if (s > sc1) {
                sc2 = sc1; ix2 = ix1;
                sc1 = s; ix1 = r;
            } else if (s > sc2) {
                sc2 = s; ix2 = r;
            }
        }
        const float w0 = hnet2_sig(sc0), w1 = hnet2_sig(sc1), w2 = hnet2_sig(sc2);
        const float denom = w0 + w1 + w2;
        active_exp[0] = 0u; active_w[0] = 1.0f;
        active_exp[1] = ix0 + 1u; active_w[1] = w0 / denom;
        active_exp[2] = ix1 + 1u; active_w[2] = w1 / denom;
        active_exp[3] = ix2 + 1u; active_w[3] = w2 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < 4u; ++e) {
        const uint exp_idx = active_exp[e];
        const float weight = active_w[e];
        if (tid < 256u) {
            const uint gu_base = exp_idx * 512u * 512u;
            float g = 0.0f, up = 0.0f;
            for (uint d = 0u; d < 512u; ++d) {
                const float x = norm_mid[d];
                const uint row_off = gu_base + d * 512u;
                g = fma(x, float(gate_up0[row_off + tid]), g);
                up = fma(x, float(gate_up0[row_off + 256u + tid]), up);
            }
            act[tid] = weight * (g * hnet2_sig(g) * up);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint dn_base = exp_idx * 256u * 512u;
        for (uint d = tid; d < 512u; d += 256u) {
            float acc = state[d];
            for (uint m = 0u; m < 256u; ++m) {
                acc = fma(act[m], float(down0[dn_base + m * 512u + d]), acc);
            }
            state[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq2 = 0.0f;
    for (uint d = tid; d < 512u; d += 256u) {
        sq2 = fma(state[d], state[d], sq2);
    }
    sq2 = simd_sum(sq2);
    if (lane == 0u) sg_sq[sg_id] = sq2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm_sq = 0.0f;
    for (uint s = 0u; s < 8u; ++s) norm_sq += sg_sq[s];
    const float inv_norm = 1.0f / max(sqrt(norm_sq), 1e-12f);
    for (uint d = tid; d < 512u; d += 256u) {
        const float u0 = float(ffn_norm[d]) * (state[d] * inv_norm);
        state[d] = u0;
        u0_table[b * 512u + d] = u0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float c0 = 0.0f, c4 = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            c0 += float(attn_ctrl[j]) + float(moe_ctrl[j]);
            c4 += float(attn_ctrl[16u + j]) + float(moe_ctrl[16u + j]);
        }
        const float ctrl0 = 0.25f * c0 + state[0] * float(attn_norm[0]) - 1.0f;
        const float ctrl4 = 0.25f * c4 + state[1] * float(attn_norm[4u * 512u + 1u]) - 1.0f;
        gates_table[b * 2u + 0u] = 0.03f + 0.06f * hnet2_sig(ctrl0);
        gates_table[b * 2u + 1u] = 0.08f + 0.16f * hnet2_sig(ctrl4);
    }
}

kernel void hnet2_prep_proj(
    device const float* u0_table [[buffer(0)]],
    device const half* hnet_w [[buffer(1)]],
    device const half* attn_norm [[buffer(2)]],
    device const half* ffn_norm [[buffer(3)]],
    device const float* scalars [[buffer(4)]],
    device float* q0_table [[buffer(5)]],
    device float* k0_table [[buffer(6)]],
    device float* r0_table [[buffer(7)]],
    device float* q1_table [[buffer(8)]],
    device float* k1_table [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, b = gid.y;
    if (d >= 512u || b >= 256u) return;
    const uint D = 512u;
    const float c_id = scalars[2];
    const device float* u0 = u0_table + b * D;
    float q0 = 0.0f, k0 = 0.0f, r0 = 0.0f, q1_raw = 0.0f, k1_raw = 0.0f;
    for (uint j = 0u; j < D; ++j) {
        const float x0 = u0[j];
        const float x1 = float(attn_norm[D + j]) * x0;
        const float wq = float(hnet_w[d * D + j]);
        const float wk = float(hnet_w[D * D + d * D + j]);
        const float wr = float(hnet_w[2u * D * D + d * D + j]);
        q0 = fma(wq, x0, q0);
        k0 = fma(wk, x0, k0);
        r0 = fma(wr, x0, r0);
        q1_raw = fma(wq, x1, q1_raw);
        k1_raw = fma(wk, x1, k1_raw);
    }
    const uint d_next1 = (d + 1u) & 511u;
    const uint d_next5 = (d + 5u) & 511u;
    const uint d_next3 = (d + 3u) & 511u;
    const uint d_next4 = (d + 4u) & 511u;
    const float u0_base_q = 0.35f * u0[d] + u0[d_next3];
    const float u0_base_k = 0.35f * u0[d] + u0[d_next4];
    const float u1_self = float(attn_norm[D + d]) * u0[d];
    const float u1_shift_q = 0.15f * u1_self + float(attn_norm[D + d_next1]) * u0[d_next1];
    const float u1_shift_k = 0.15f * u1_self + float(attn_norm[D + d_next5]) * u0[d_next5];
    const float fn1 = float(ffn_norm[D + d]);
    q0_table[b * D + d] = 1.00f * q0 + c_id * (u0_base_q - 1.00f * u0[d]);
    k0_table[b * D + d] = 1.00f * k0 + c_id * (u0_base_k - 1.00f * u0[d]);
    r0_table[b * D + d] = r0;
    q1_table[b * D + d] = fn1 * (0.35f * q1_raw + c_id * (u1_shift_q - 0.35f * u1_self));
    k1_table[b * D + d] = fn1 * (0.35f * k1_raw + c_id * (u1_shift_k - 0.35f * u1_self));
}

kernel void hnet2_prep_readout(
    device const float* r0_table [[buffer(0)]],
    device const half* hnet_w [[buffer(1)]],
    device const half* ffn_norm [[buffer(2)]],
    device const half* final_norm [[buffer(3)]],
    device const half* readout [[buffer(4)]],
    device float* r0_logits [[buffer(5)]],
    device float* b_logits [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint v = gid.x, row = gid.y;
    if (v >= 256u || row >= 257u) return;
    const uint D = 512u;
    float sum = 0.0f;
    if (row < 256u) {
        const device float* r0 = r0_table + row * D;
        for (uint d = 0u; d < D; ++d) {
            const float scale = float(final_norm[d]) * float(ffn_norm[4u * D + d]) * r0[d];
            sum = fma(scale, float(readout[d * 256u + v]), sum);
        }
        r0_logits[row * 256u + v] = sum;
    } else {
        for (uint d = 0u; d < D; ++d) {
            const float b_res = float(hnet_w[3u * D * D + d]);
            const float scale = float(final_norm[d]) * float(ffn_norm[4u * D + d]) * b_res;
            sum = fma(scale, float(readout[d * 256u + v]), sum);
        }
        b_logits[v] = sum;
    }
}

kernel void hnet2_stage0_reduce_tile(
    device const uint* tokens [[buffer(0)]],
    device const float* u0_table [[buffer(1)]],
    device const float* q0_table [[buffer(2)]],
    device const float* k0_table [[buffer(3)]],
    device const float* q1_table [[buffer(4)]],
    device const float* k1_table [[buffer(5)]],
    device const float* gates_table [[buffer(6)]],
    device float* s0_blk_a [[buffer(7)]],
    device float* s0_blk_b [[buffer(8)]],
    constant HNetShape& shape [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, blk = gid.y;
    if (d >= 512u || blk >= 256u) return;
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint t_start = min(blk * 128u, L);
    const uint t_end = min(t_start + 128u, L);
    const uint t_tail = t_end > t_start + 16u ? (t_end - 16u) : t_start;
    float a = t_tail > t_start ? 0.0f : 1.0f;
    float bh = 0.0f, bq0 = 0.0f, bk0 = 0.0f, bq1 = 0.0f, bk1 = 0.0f;
    for (uint t = t_tail; t < t_end; ++t) {
        const uint b = tokens[tile_off + t] & 255u;
        const float alpha = gates_table[b * 2u];
        const float om = 1.0f - alpha;
        const uint idx = b * 512u + d;
        a *= alpha;
        bh = fma(alpha, bh, om * u0_table[idx]);
        bq0 = fma(alpha, bq0, om * q0_table[idx]);
        bk0 = fma(alpha, bk0, om * k0_table[idx]);
        bq1 = fma(alpha, bq1, om * q1_table[idx]);
        bk1 = fma(alpha, bk1, om * k1_table[idx]);
    }
    const uint base = blk * 5u * 512u + d;
    s0_blk_b[base + 0u * 512u] = bh;
    s0_blk_b[base + 1u * 512u] = bq0;
    s0_blk_b[base + 2u * 512u] = bk0;
    s0_blk_b[base + 3u * 512u] = bq1;
    s0_blk_b[base + 4u * 512u] = bk1;
    if (d == 0u) s0_blk_a[blk] = a;
}

kernel void hnet2_stage0_route_tile(
    device const uint* tokens [[buffer(0)]],
    device const float* q0_table [[buffer(1)]],
    device const float* k0_table [[buffer(2)]],
    device const float* gates_table [[buffer(3)]],
    device const float* s0_blk_a [[buffer(4)]],
    device const float* s0_blk_b [[buffer(5)]],
    device const float* q0_state [[buffer(6)]],
    device const float* k0_state [[buffer(7)]],
    device const uint* seen0 [[buffer(8)]],
    device float* probs0 [[buffer(9)]],
    constant HNetShape& shape [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    (void)s0_blk_a;
    (void)s0_blk_b;
    (void)q0_state;
    (void)k0_state;
    const uint blk = tgid * 8u + sg_id;
    if (blk >= 512u) return;
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint t_start = min(blk * 64u, L);
    const uint t_end = min(t_start + 64u, L);
    if (t_start >= t_end) return;

    float q[16], k[16];
    for (uint i = 0u; i < 16u; ++i) {
        q[i] = 0.0f;
        k[i] = 0.0f;
    }

    const uint g_start = tile_off + t_start;
    const uint g_warm = g_start > 16u ? (g_start - 16u) : 0u;
    for (uint g = g_warm; g < g_start; ++g) {
        const uint b = tokens[g] & 255u;
        const float alpha = gates_table[b * 2u];
        const float om = 1.0f - alpha;
        const uint b_off = b * 512u + lane;
        for (uint i = 0u; i < 16u; ++i) {
            k[i] = fma(alpha, k[i], om * k0_table[b_off + 32u * i]);
            q[i] = fma(alpha, q[i], om * q0_table[b_off + 32u * i]);
        }
    }

    const bool first_tile = (seen0[0] == 0u);
    for (uint t = t_start; t < t_end; ++t) {
        const uint b = tokens[tile_off + t] & 255u;
        const float alpha = gates_table[b * 2u];
        const float om = 1.0f - alpha;
        const uint b_off = b * 512u + lane;
        float dot = 0.0f, nq = 0.0f, nk = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            const float ki = fma(alpha, k[i], om * k0_table[b_off + 32u * i]);
            k[i] = ki;
            const float qi = q[i];
            dot = fma(qi, ki, dot);
            nq = fma(qi, qi, nq);
            nk = fma(ki, ki, nk);
            q[i] = fma(alpha, qi, om * q0_table[b_off + 32u * i]);
        }
        const float denom = max(sqrt(simd_sum(nq)), 1e-12f) * max(sqrt(simd_sum(nk)), 1e-12f);
        const float cos_val = simd_sum(dot) / denom;
        const bool is_first = (t == 0u) && first_tile;
        if (lane == 0u) {
            probs0[tile_off + t] = is_first ? 1.0f : clamp((1.0f - cos_val) * 0.5f, 0.0f, 1.0f);
        }
    }
}

kernel void hnet2_stage0_scan_tile(
    device const float* probs0 [[buffer(0)]],
    device const uint* count0 [[buffer(1)]],
    device const uint* seen0 [[buffer(2)]],
    device uint* origins0 [[buffer(3)]],
    device uint* plug0 [[buffer(4)]],
    device uint* tile_m0 [[buffer(5)]],
    constant HNetShape& shape [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint c0_start = count0[0];
    const bool first_is_b = (seen0[0] == 0u);
    threadgroup uint sg_totals[32];
    uint running_base = 0u;
    for (uint base = 0u; base < L; base += 1024u) {
        const uint t = base + tid;
        const uint is_b = (t < L && ((t == 0u && first_is_b) || probs0[tile_off + t] > 0.5f)) ? 1u : 0u;
        const uint sg_prefix = simd_prefix_exclusive_sum(is_b);
        const uint sg_sum = simd_sum(is_b);
        if (lane == 0u) sg_totals[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint s_val = sg_totals[lane];
        const uint sg_offset = simd_broadcast(simd_prefix_exclusive_sum(s_val), sg_id);
        const uint block_total = simd_sum(s_val);
        const uint local_idx = running_base + sg_offset + sg_prefix;
        if (t < L) {
            const uint global_c0 = c0_start + local_idx;
            if (is_b != 0u) origins0[global_c0] = tile_off + t;
            plug0[tile_off + t] = (global_c0 + is_b) - 1u;
        }
        running_base += block_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) tile_m0[0] = running_base;
}

kernel void hnet2_stage0_pack_tile(
    device const uint* tokens [[buffer(0)]],
    device const float* u0_table [[buffer(1)]],
    device const float* q1_table [[buffer(2)]],
    device const float* k1_table [[buffer(3)]],
    device const float* gates_table [[buffer(4)]],
    device const float* s0_blk_a [[buffer(5)]],
    device const float* s0_blk_b [[buffer(6)]],
    device const float* h0_state [[buffer(7)]],
    device const float* q1s_state [[buffer(8)]],
    device const float* k1s_state [[buffer(9)]],
    device const float* probs0 [[buffer(10)]],
    device const uint* plug0 [[buffer(11)]],
    device const uint* count0 [[buffer(12)]],
    device const uint* seen0 [[buffer(13)]],
    device float* tile_z0 [[buffer(14)]],
    device float* tile_q1s [[buffer(15)]],
    device float* tile_k1s [[buffer(16)]],
    constant HNetShape& shape [[buffer(17)]],
    uint2 gid [[thread_position_in_grid]]) {
    (void)s0_blk_a;
    (void)s0_blk_b;
    (void)h0_state;
    (void)q1s_state;
    (void)k1s_state;
    const uint d = gid.x, blk = gid.y;
    if (d >= 512u || blk >= 256u) return;
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint t_start = min(blk * 128u, L);
    const uint t_end = min(t_start + 128u, L);
    if (t_start >= t_end) return;

    float h0 = 0.0f, q1s = 0.0f, k1s = 0.0f;
    const uint g_start = tile_off + t_start;
    const uint g_warm = g_start > 16u ? (g_start - 16u) : 0u;
    for (uint g = g_warm; g < g_start; ++g) {
        const uint b = tokens[g] & 255u;
        const float alpha = gates_table[b * 2u];
        const float om = 1.0f - alpha;
        const uint idx = b * 512u + d;
        h0 = fma(alpha, h0, om * u0_table[idx]);
        q1s = fma(alpha, q1s, om * q1_table[idx]);
        k1s = fma(alpha, k1s, om * k1_table[idx]);
    }

    const uint c0_start = count0[0];
    const bool first_is_b = (seen0[0] == 0u);
    for (uint t = t_start; t < t_end; ++t) {
        const uint b = tokens[tile_off + t] & 255u;
        const float alpha = gates_table[b * 2u];
        const float om = 1.0f - alpha;
        const uint idx = b * 512u + d;
        h0 = fma(alpha, h0, om * u0_table[idx]);
        q1s = fma(alpha, q1s, om * q1_table[idx]);
        k1s = fma(alpha, k1s, om * k1_table[idx]);
        if ((t == 0u && first_is_b) || probs0[tile_off + t] > 0.5f) {
            const uint c_local = plug0[tile_off + t] - c0_start;
            tile_z0[c_local * 512u + d] = h0;
            tile_q1s[c_local * 512u + d] = q1s;
            tile_k1s[c_local * 512u + d] = k1s;
        }
    }
}

kernel void hnet2_stage1_reduce_tile(
    device const float* tile_z0 [[buffer(0)]],
    device const float* tile_q1s [[buffer(1)]],
    device const float* tile_k1s [[buffer(2)]],
    device const half* attn_norm [[buffer(3)]],
    device const float* scalars [[buffer(4)]],
    device const uint* tile_m0 [[buffer(5)]],
    device float* s1_blk_a [[buffer(6)]],
    device float* s1_blk_b [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, blk = gid.y;
    if (d >= 512u || blk >= 256u) return;
    const uint m0 = tile_m0[0];
    const uint c_start = min(blk * 128u, m0);
    const uint c_end = min(c_start + 128u, m0);
    if (c_start >= c_end) return;
    const uint c_tail = c_end > c_start + 16u ? (c_end - 16u) : c_start;
    const float g1 = scalars[0];
    const float om = 1.0f - g1;
    const float an1 = float(attn_norm[512u + d]);
    float a = c_tail > c_start ? 0.0f : 1.0f;
    float bh1 = 0.0f, bq1 = 0.0f, bk1 = 0.0f;
    for (uint c = c_tail; c < c_end; ++c) {
        const uint idx = c * 512u + d;
        a *= g1;
        bh1 = fma(g1, bh1, om * an1 * tile_z0[idx]);
        bq1 = fma(g1, bq1, om * tile_q1s[idx]);
        bk1 = fma(g1, bk1, om * tile_k1s[idx]);
    }
    const uint base = blk * 3u * 512u + d;
    s1_blk_b[base + 0u * 512u] = bh1;
    s1_blk_b[base + 1u * 512u] = bq1;
    s1_blk_b[base + 2u * 512u] = bk1;
    if (d == 0u) s1_blk_a[blk] = a;
}

kernel void hnet2_stage1_route_tile(
    device const float* tile_q1s [[buffer(0)]],
    device const float* tile_k1s [[buffer(1)]],
    device const float* scalars [[buffer(2)]],
    device const float* s1_blk_a [[buffer(3)]],
    device const float* s1_blk_b [[buffer(4)]],
    device const float* q1_state [[buffer(5)]],
    device const float* k1_state [[buffer(6)]],
    device const uint* count0 [[buffer(7)]],
    device const uint* tile_m0 [[buffer(8)]],
    device const uint* seen1 [[buffer(9)]],
    device float* probs1 [[buffer(10)]],
    constant HNetShape& shape [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    (void)s1_blk_a;
    (void)q1_state;
    (void)k1_state;
    const uint blk = tgid * 8u + sg_id;
    if (blk >= 256u) return;
    const uint m0 = tile_m0[0];
    const uint c_start = min(blk * 128u, m0);
    const uint c_end = min(c_start + 128u, m0);
    if (c_start >= c_end) return;

    float q[16], k[16];
    const uint base = blk * 3u * 512u + lane;
    for (uint i = 0u; i < 16u; ++i) {
        q[i] = s1_blk_b[base + 1u * 512u + 32u * i];
        k[i] = s1_blk_b[base + 2u * 512u + 32u * i];
    }

    const float g1 = scalars[0];
    const float om = 1.0f - g1;
    const uint c0_start = count0[0];
    const uint max_c0 = shape.max_chunks;
    const bool first_tile = (seen1[0] == 0u);
    for (uint c = c_start; c < c_end; ++c) {
        const uint c_off = c * 512u + lane;
        float dot = 0.0f, nq = 0.0f, nk = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            const float ki = fma(g1, k[i], om * tile_k1s[c_off + 32u * i]);
            k[i] = ki;
            const float qi = q[i];
            dot = fma(qi, ki, dot);
            nq = fma(qi, qi, nq);
            nk = fma(ki, ki, nk);
            q[i] = fma(g1, qi, om * tile_q1s[c_off + 32u * i]);
        }
        const float denom = max(sqrt(simd_sum(nq)), 1e-12f) * max(sqrt(simd_sum(nk)), 1e-12f);
        const float cos_val = simd_sum(dot) / denom;
        const bool is_first = (c == 0u) && first_tile;
        if (lane == 0u) {
            const uint dst = min(c0_start + c, max_c0 - 1u);
            probs1[dst] = is_first ? 1.0f : clamp((1.0f - cos_val) * 0.5f, 0.0f, 1.0f);
        }
    }
}

kernel void hnet2_stage1_scan_tile(
    device const float* probs1 [[buffer(0)]],
    device const uint* count0 [[buffer(1)]],
    device const uint* tile_m0 [[buffer(2)]],
    device const uint* count1 [[buffer(3)]],
    device const uint* seen1 [[buffer(4)]],
    device uint* origins1 [[buffer(5)]],
    device uint* plug1 [[buffer(6)]],
    device uint* tile_m1 [[buffer(7)]],
    constant HNetShape& shape [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint m0 = tile_m0[0];
    const uint c0_start = count0[0];
    const uint c1_start = count1[0];
    const uint max_c0 = shape.max_chunks;
    const uint max_c1 = shape.vocab_size;
    const bool first_is_b = (seen1[0] == 0u);
    threadgroup uint sg_totals[32];
    uint running_base = 0u;
    for (uint base = 0u; base < m0; base += 1024u) {
        const uint c = base + tid;
        const uint g_c0 = min(c0_start + c, max_c0 - 1u);
        const uint is_b = (c < m0 && ((c == 0u && first_is_b) || probs1[g_c0] > 0.5f)) ? 1u : 0u;
        const uint sg_prefix = simd_prefix_exclusive_sum(is_b);
        const uint sg_sum = simd_sum(is_b);
        if (lane == 0u) sg_totals[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint s_val = sg_totals[lane];
        const uint sg_offset = simd_broadcast(simd_prefix_exclusive_sum(s_val), sg_id);
        const uint block_total = simd_sum(s_val);
        const uint local_idx = running_base + sg_offset + sg_prefix;
        if (c < m0) {
            const uint g_c1 = min(c1_start + local_idx, max_c1 - 1u);
            if (is_b != 0u) origins1[g_c1] = g_c0;
            plug1[g_c0] = (c1_start + local_idx + is_b) - 1u;
        }
        running_base += block_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) tile_m1[0] = running_base;
}

kernel void hnet2_stage1_pack_tile(
    device const float* tile_z0 [[buffer(0)]],
    device const half* attn_norm [[buffer(1)]],
    device const float* scalars [[buffer(2)]],
    device const float* s1_blk_a [[buffer(3)]],
    device const float* s1_blk_b [[buffer(4)]],
    device const float* h1_state [[buffer(5)]],
    device const float* probs1 [[buffer(6)]],
    device const uint* plug1 [[buffer(7)]],
    device const uint* count0 [[buffer(8)]],
    device const uint* tile_m0 [[buffer(9)]],
    device const uint* seen1 [[buffer(10)]],
    device float* chunks_h0 [[buffer(11)]],
    device half* chunks_h1 [[buffer(12)]],
    constant HNetShape& shape [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]) {
    (void)s1_blk_a;
    (void)h1_state;
    const uint d = gid.x, blk = gid.y;
    if (d >= 512u || blk >= 256u) return;
    const uint m0 = tile_m0[0];
    const uint c_start = min(blk * 128u, m0);
    const uint c_end = min(c_start + 128u, m0);
    if (c_start >= c_end) return;

    float h1 = s1_blk_b[(blk * 3u + 0u) * 512u + d];
    const float g1 = scalars[0];
    const float om = 1.0f - g1;
    const float an1 = float(attn_norm[512u + d]);
    const uint c0_start = count0[0];
    const uint max_c0 = shape.max_chunks;
    const uint max_c1 = shape.vocab_size;
    const bool first_is_b = (seen1[0] == 0u);
    for (uint c = c_start; c < c_end; ++c) {
        h1 = fma(g1, h1, om * an1 * tile_z0[c * 512u + d]);
        const uint g_c0 = min(c0_start + c, max_c0 - 1u);
        chunks_h0[g_c0 * 512u + d] = h1;
        if ((c == 0u && first_is_b) || probs1[g_c0] > 0.5f) {
            const uint g_c1 = min(plug1[g_c0], max_c1 - 1u);
            chunks_h1[g_c1 * 512u + d] = half(h1);
        }
    }
}

kernel void hnet2_tile_advance(
    device const float* s0_blk_a [[buffer(0)]],
    device float* s0_blk_b [[buffer(1)]],
    device const float* s1_blk_a [[buffer(2)]],
    device float* s1_blk_b [[buffer(3)]],
    device float* h0_state [[buffer(4)]],
    device float* q0_state [[buffer(5)]],
    device float* k0_state [[buffer(6)]],
    device float* q1s_state [[buffer(7)]],
    device float* k1s_state [[buffer(8)]],
    device float* h1_state [[buffer(9)]],
    device float* q1_state [[buffer(10)]],
    device float* k1_state [[buffer(11)]],
    device uint* count0 [[buffer(12)]],
    device const uint* tile_m0 [[buffer(13)]],
    device uint* count1 [[buffer(14)]],
    device const uint* tile_m1 [[buffer(15)]],
    device uint* seen0 [[buffer(16)]],
    device uint* seen1 [[buffer(17)]],
    constant HNetShape& shape [[buffer(18)]],
    uint d [[thread_index_in_threadgroup]]) {
    if (d >= 512u) return;
    if (shape.score == 0u) {
        const uint n_blk0 = min((shape.seq_len + 127u) / 128u, 256u);
        float h0 = h0_state[d], q0 = q0_state[d], k0 = k0_state[d];
        float q1s = q1s_state[d], k1s = k1s_state[d];
        for (uint p = 0u; p < n_blk0; ++p) {
            const float ap = s0_blk_a[p];
            const uint base = p * 5u * 512u + d;
            const float bh = s0_blk_b[base + 0u * 512u];
            const float bq0 = s0_blk_b[base + 1u * 512u];
            const float bk0 = s0_blk_b[base + 2u * 512u];
            const float bq1 = s0_blk_b[base + 3u * 512u];
            const float bk1 = s0_blk_b[base + 4u * 512u];
            s0_blk_b[base + 0u * 512u] = h0;
            s0_blk_b[base + 1u * 512u] = q0;
            s0_blk_b[base + 2u * 512u] = k0;
            s0_blk_b[base + 3u * 512u] = q1s;
            s0_blk_b[base + 4u * 512u] = k1s;
            h0 = fma(ap, h0, bh);
            q0 = fma(ap, q0, bq0);
            k0 = fma(ap, k0, bk0);
            q1s = fma(ap, q1s, bq1);
            k1s = fma(ap, k1s, bk1);
        }
        h0_state[d] = h0;
        q0_state[d] = q0;
        k0_state[d] = k0;
        q1s_state[d] = q1s;
        k1s_state[d] = k1s;
        return;
    }
    if (shape.score == 1u) {
        const uint n_blk1 = min((tile_m0[0] + 127u) / 128u, 256u);
        float h1 = h1_state[d], q1 = q1_state[d], k1 = k1_state[d];
        for (uint p = 0u; p < n_blk1; ++p) {
            const float ap = s1_blk_a[p];
            const uint base = p * 3u * 512u + d;
            const float bh1 = s1_blk_b[base + 0u * 512u];
            const float bq1 = s1_blk_b[base + 1u * 512u];
            const float bk1 = s1_blk_b[base + 2u * 512u];
            s1_blk_b[base + 0u * 512u] = h1;
            s1_blk_b[base + 1u * 512u] = q1;
            s1_blk_b[base + 2u * 512u] = k1;
            h1 = fma(ap, h1, bh1);
            q1 = fma(ap, q1, bq1);
            k1 = fma(ap, k1, bk1);
        }
        h1_state[d] = h1;
        q1_state[d] = q1;
        k1_state[d] = k1;
        return;
    }
    if (d == 0u) {
        const uint m0 = tile_m0[0];
        const uint m1 = tile_m1[0];
        count0[0] += m0;
        count1[0] += m1;
        seen0[0] = 1u;
        if (m0 > 0u) seen1[0] = 1u;
    }
}

kernel void hnet2_stage2_load(
    device const half* chunks_h1 [[buffer(0)]],
    device half* streams [[buffer(1)]],
    constant HNetShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model, rows = shape.max_chunks, valid = shape.seq_len;
    const uint stride = max(shape.vocab_size, 1u);
    const uint total_m1 = shape.batch_size;
    if (index >= rows * D) return;
    const uint r = index / D, d = index % D;
    half val = half(0.0f);
    if (r < valid) {
        if (stride == 1u) {
            val = chunks_h1[r * D + d];
        } else {
            const uint c_start = r * stride;
            const uint c_end = min(c_start + stride, total_m1);
            float acc = float(chunks_h1[c_start * D + d]);
            for (uint c = c_start + 1u; c < c_end; ++c) {
                acc = fma(0.85f, acc, 0.15f * float(chunks_h1[c * D + d]));
            }
            val = half(acc);
        }
    }
    for (uint s = 0u; s < 4u; ++s) {
        streams[(r * 4u + s) * D + d] = val;
    }
}

kernel void hnet2_stage2_store(
    device const half* normalized [[buffer(0)]],
    device const half* chunks_h1 [[buffer(1)]],
    device half* core_h1 [[buffer(2)]],
    constant HNetShape& shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model, total_m1 = shape.batch_size;
    const uint stride = max(shape.vocab_size, 1u);
    if (index >= total_m1 * D) return;
    if (stride == 1u) {
        core_h1[index] = normalized[index];
    } else {
        const uint c1 = index / D, d = index % D;
        const uint g = c1 / stride;
        const float ctx = g > 0u ? float(normalized[(g - 1u) * D + d]) : float(chunks_h1[d]);
        core_h1[index] = half(0.5f * ctx + 0.5f * float(chunks_h1[index]));
    }
}

kernel void hnet2_dechunk1_reduce(
    device const half* core_h1 [[buffer(0)]],
    device float* ema1 [[buffer(1)]],
    device const uint* origins1 [[buffer(2)]],
    device const float* probs1 [[buffer(3)]],
    device const uint* count1 [[buffer(4)]],
    device float* part_a [[buffer(5)]],
    device float* part_b [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, part = gid.y;
    if (d >= 512u || part >= 128u) return;
    const uint total = min(count1[0], shape.vocab_size);
    const uint part_size = max((total + 127u) / 128u, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    if (c_start >= c_end) {
        part_a[part * 512u + d] = 1.0f;
        part_b[part * 512u + d] = 0.0f;
        return;
    }
    (void)ema1;
    const uint c_tail = c_end > c_start + 40u ? (c_end - 40u) : c_start;
    float a = c_tail > c_start ? 0.0f : 1.0f;
    float b = 0.0f;
    for (uint c = c_tail; c < c_end; ++c) {
        const float p = clamp(probs1[origins1[c]], 1e-4f, 1.0f - 1e-4f);
        a = (1.0f - p) * a;
        b = fma(p, float(core_h1[c * 512u + d]), (1.0f - p) * b);
    }
    part_a[part * 512u + d] = a;
    part_b[part * 512u + d] = b;
}

kernel void hnet2_dechunk1_scan(
    device const half* core_h1 [[buffer(0)]],
    device float* ema1 [[buffer(1)]],
    device const uint* origins1 [[buffer(2)]],
    device const float* probs1 [[buffer(3)]],
    device const uint* count1 [[buffer(4)]],
    device const float* part_a [[buffer(5)]],
    device const float* part_b [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    (void)part_a;
    (void)part_b;
    const uint d = gid.x, part = gid.y;
    if (d >= 512u || part >= 1024u) return;
    const uint total = min(count1[0], shape.vocab_size);
    const uint part_size = max((total + 1023u) / 1024u, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    if (c_start >= c_end) return;
    float state = 0.0f;
    const uint c_warm = c_start > 36u ? (c_start - 36u) : 0u;
    for (uint c = c_warm; c < c_start; ++c) {
        const float p = clamp(probs1[origins1[c]], 1e-4f, 1.0f - 1e-4f);
        state = fma(p, float(core_h1[c * 512u + d]), (1.0f - p) * state);
    }
    for (uint c = c_start; c < c_end; ++c) {
        const float p = clamp(probs1[origins1[c]], 1e-4f, 1.0f - 1e-4f);
        state = fma(p, float(core_h1[c * 512u + d]), (1.0f - p) * state);
        ema1[c * 512u + d] = state;
    }
}

kernel void hnet2_dechunk0_reduce(
    device const float* ema1 [[buffer(0)]],
    device const float* chunks_h0 [[buffer(1)]],
    device const uint* plug1 [[buffer(2)]],
    device const uint* origins0 [[buffer(3)]],
    device const float* probs0 [[buffer(4)]],
    device const uint* count0 [[buffer(5)]],
    device const half* attn_norm [[buffer(6)]],
    device const half* ffn_norm [[buffer(7)]],
    device const float* scalars [[buffer(8)]],
    device float* d0_part [[buffer(9)]],
    constant HNetShape& shape [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, part = gid.y;
    if (d >= 512u || part >= 1024u) return;
    const uint total = min(count0[0], shape.max_chunks);
    const uint max_c1 = shape.vocab_size;
    const uint part_size = max((total + 1023u) / 1024u, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    const float g3 = scalars[1];
    const float om_g3 = 1.0f - g3;
    const float an3 = float(attn_norm[3u * 512u + d]);
    const float fn3 = float(ffn_norm[3u * 512u + d]);

    const uint c_tail = c_end > c_start + 36u ? (c_end - 36u) : c_start;
    float a = c_tail > c_start ? 0.0f : 1.0f;
    float b = c_tail > c_start ? 0.0f : 1.0f;
    float c_coef = 0.0f, u_acc = 0.0f, v_acc = 0.0f;
    for (uint c0 = c_tail; c0 < c_end; ++c0) {
        const uint c1 = min(plug1[c0], max_c1 - 1u);
        const float e1 = ema1[c1 * 512u + d];
        const float h1 = chunks_h0[c0 * 512u + d];
        const float u = om_g3 * fn3 * (e1 + an3 * h1);
        const float p0 = clamp(probs0[origins0[c0]], 1e-4f, 1.0f - 1e-4f);
        const float b2 = 1.0f - p0;
        const float c2 = p0 * g3;
        const float v = p0 * u;

        const float next_a = g3 * a;
        const float next_u = fma(g3, u_acc, u);
        const float next_c = fma(c2, a, b2 * c_coef);
        const float next_b = b2 * b;
        const float next_v = fma(c2, u_acc, fma(b2, v_acc, v));
        a = next_a; b = next_b; c_coef = next_c; u_acc = next_u; v_acc = next_v;
    }
    const uint base = part * 5u * 512u + d;
    d0_part[base + 0u * 512u] = a;
    d0_part[base + 1u * 512u] = b;
    d0_part[base + 2u * 512u] = c_coef;
    d0_part[base + 3u * 512u] = u_acc;
    d0_part[base + 4u * 512u] = v_acc;
}

kernel void hnet2_dechunk0_scan(
    device const float* ema1 [[buffer(0)]],
    device float* chunks_h0 [[buffer(1)]],
    device const uint* plug1 [[buffer(2)]],
    device const uint* origins0 [[buffer(3)]],
    device const float* probs0 [[buffer(4)]],
    device const uint* count0 [[buffer(5)]],
    device const half* attn_norm [[buffer(6)]],
    device const half* ffn_norm [[buffer(7)]],
    device const float* scalars [[buffer(8)]],
    device const float* d0_part [[buffer(9)]],
    device float* saved_h1 [[buffer(10)]],
    constant HNetShape& shape [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, part = gid.y;
    if (d >= 512u || part >= 1024u) return;
    const uint total = min(count0[0], shape.max_chunks);
    const uint max_c1 = shape.vocab_size;
    const uint part_size = max((total + 1023u) / 1024u, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    if (c_start >= c_end) return;

    float y1 = 0.0f, ema0 = 0.0f;
    const uint k_start = (part_size >= 36u && part > 0u)
        ? (part - 1u)
        : (part > 36u ? (part - 36u) : 0u);
    for (uint k = k_start; k < part; ++k) {
        const uint base = k * 5u * 512u + d;
        const float ap = d0_part[base + 0u * 512u];
        const float bp = d0_part[base + 1u * 512u];
        const float cp = d0_part[base + 2u * 512u];
        const float up = d0_part[base + 3u * 512u];
        const float vp = d0_part[base + 4u * 512u];
        const float next_y1 = fma(ap, y1, up);
        const float next_e0 = fma(cp, y1, fma(bp, ema0, vp));
        y1 = next_y1; ema0 = next_e0;
    }

    const float g3 = scalars[1];
    const float om_g3 = 1.0f - g3;
    const float an3 = float(attn_norm[3u * 512u + d]);
    const float fn3 = float(ffn_norm[3u * 512u + d]);
    uint cur_c1 = 0xffffffffu;
    float e1 = 0.0f;
    for (uint c0 = c_start; c0 < c_end; ++c0) {
        const uint c1 = min(plug1[c0], max_c1 - 1u);
        if (c1 != cur_c1) {
            cur_c1 = c1;
            e1 = ema1[c1 * 512u + d];
        }
        const float h1 = chunks_h0[c0 * 512u + d];
        if (c0 < 4096u) saved_h1[c0 * 512u + d] = h1;
        const float z1 = fn3 * (e1 + an3 * h1);
        y1 = fma(g3, y1, om_g3 * z1);
        const float p0 = clamp(probs0[origins0[c0]], 1e-4f, 1.0f - 1e-4f);
        ema0 = fma(p0, y1, (1.0f - p0) * ema0);
        chunks_h0[c0 * 512u + d] = ema0;
    }
}

kernel void hnet2_ema0_logits_tile(
    device const float* ema0 [[buffer(0)]],
    device const uint* plug0 [[buffer(1)]],
    device const half* ffn_norm [[buffer(2)]],
    device const half* final_norm [[buffer(3)]],
    device const half* readout [[buffer(4)]],
    device const float* b_logits [[buffer(5)]],
    device float* tile_ema0_logits [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]]) {
    const uint L = shape.seq_len;
    if (L == 0u) return;
    const uint tile_off = shape.batch_size;
    const uint max_c0 = shape.max_chunks;
    const uint warm_off = tile_off > 20u ? (tile_off - 20u) : 0u;
    const uint c0_base = plug0[warm_off];
    const uint c0_last = plug0[tile_off + L - 1u];
    const uint c_count = (c0_last - c0_base) + 1u;

    const uint col_base = (tgid & 3u) * 64u;
    const uint row_tg = tgid >> 2u;
    const uint sg_r = sg_id >> 1u;
    const uint sg_c = sg_id & 1u;
    const uint r_load = tid >> 3u;
    const uint k_load = tid & 7u;
    const uint b_r0 = tid >> 6u;
    const uint b_c = tid & 63u;

    threadgroup float a_tile[32 * 8];
    threadgroup float b_tile[8 * 64];

    const float b_init = b_logits[col_base + b_c];
    b_tile[tid] = b_init;
    b_tile[tid + 256u] = b_init;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 b_bias[4];
    for (uint j = 0u; j < 4u; ++j) {
        simdgroup_load(b_bias[j], &b_tile[sg_c * 32u + j * 8u], 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx_base = row_tg * 32u; idx_base < c_count; idx_base += 2048u) {
        simdgroup_float8x8 acc[4];
        for (uint j = 0u; j < 4u; ++j) {
            acc[j] = b_bias[j];
        }
        const uint c0_load = min(c0_base + min(idx_base + r_load, c_count - 1u), max_c0 - 1u);
        for (uint k_base = 0u; k_base < 512u; k_base += 8u) {
            const uint d = k_base + k_load;
            a_tile[tid] = ema0[c0_load * 512u + d]
                * float(final_norm[d])
                * float(ffn_norm[4u * 512u + d]);
            b_tile[tid] = float(readout[(k_base + b_r0) * 256u + col_base + b_c]);
            b_tile[tid + 256u] = float(readout[(k_base + b_r0 + 4u) * 256u + col_base + b_c]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 a_frag;
            simdgroup_load(a_frag, &a_tile[sg_r * 64u], 8);
            for (uint j = 0u; j < 4u; ++j) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, &b_tile[sg_c * 32u + j * 8u], 64);
                simdgroup_multiply_accumulate(acc[j], a_frag, b_frag, acc[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint out_row = idx_base + sg_r * 8u;
        const uint out_col = col_base + sg_c * 32u;
        for (uint j = 0u; j < 4u; ++j) {
            simdgroup_store(
                acc[j],
                tile_ema0_logits + out_row * 256u + out_col + j * 8u,
                256);
        }
    }
}

kernel void hnet2_readout_reduce_tile(
    device const uint* tokens [[buffer(0)]],
    device const uint* plug0 [[buffer(1)]],
    device const float* gates_table [[buffer(2)]],
    device const float* r0_logits [[buffer(3)]],
    device const float* b_logits [[buffer(4)]],
    device const float* tile_ema0_logits [[buffer(5)]],
    device float* ro_blk_abc [[buffer(6)]],
    device float* ro_blk_uv [[buffer(7)]],
    constant HNetShape& shape [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
    (void)b_logits;
    const uint v = gid.x, blk = gid.y;
    if (v >= 256u || blk >= 256u) return;
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint t_start = min(blk * 128u, L);
    const uint t_end = min(t_start + 128u, L);
    if (t_start >= t_end) return;
    const uint t_tail = t_end > t_start + 20u ? (t_end - 20u) : t_start;
    const uint c0_base = L > 0u ? plug0[tile_off] : 0u;

    float a_acc = t_tail > t_start ? 0.0f : 1.0f;
    float b_acc = t_tail > t_start ? 0.0f : 1.0f;
    float c_acc = 0.0f, u_acc = 0.0f, v_acc = 0.0f;
    for (uint t = t_tail; t < t_end; ++t) {
        const uint b = tokens[tile_off + t] & 255u;
        const float alpha = gates_table[b * 2u];
        const float beta = gates_table[b * 2u + 1u];
        const float om_a = 1.0f - alpha;
        const float om_b = 1.0f - beta;
        const float u = om_a * r0_logits[b * 256u + v];
        const uint e_idx = plug0[tile_off + t] - c0_base;
        const float e = tile_ema0_logits[e_idx * 256u + v];
        const float c2 = om_b * alpha;
        const float v_step = om_b * (u + e);

        const float next_a = alpha * a_acc;
        const float next_u = fma(alpha, u_acc, u);
        const float next_c = fma(c2, a_acc, beta * c_acc);
        const float next_b = beta * b_acc;
        const float next_v = fma(c2, u_acc, fma(beta, v_acc, v_step));
        a_acc = next_a; b_acc = next_b; c_acc = next_c; u_acc = next_u; v_acc = next_v;
    }
    ro_blk_uv[(blk * 2u + 0u) * 256u + v] = u_acc;
    ro_blk_uv[(blk * 2u + 1u) * 256u + v] = v_acc;
    if (v == 0u) {
        ro_blk_abc[blk * 3u + 0u] = a_acc;
        ro_blk_abc[blk * 3u + 1u] = b_acc;
        ro_blk_abc[blk * 3u + 2u] = c_acc;
    }
}

kernel void hnet2_readout_scan_tile(
    device const uint* tokens [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device const uint* plug0 [[buffer(2)]],
    device const float* gates_table [[buffer(3)]],
    device const float* r0_logits [[buffer(4)]],
    device const float* b_logits [[buffer(5)]],
    device const float* tile_ema0_logits [[buffer(6)]],
    device const float* ro_blk_abc [[buffer(7)]],
    device const float* ro_blk_uv [[buffer(8)]],
    device const float* r_logit_state [[buffer(9)]],
    device const float* y0_logit_state [[buffer(10)]],
    device uint* predictions [[buffer(11)]],
    device float* losses [[buffer(12)]],
    constant HNetShape& shape [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    (void)b_logits;
    (void)ro_blk_abc;
    (void)ro_blk_uv;
    (void)r_logit_state;
    (void)y0_logit_state;
    const uint blk = tgid * 8u + sg_id;
    if (blk >= 512u) return;
    const uint L = shape.seq_len;
    const uint tile_off = shape.batch_size;
    const uint t_start = min(blk * 64u, L);
    const uint t_end = min(t_start + 64u, L);
    if (t_start >= t_end) return;

    float r_hat[8], ell[8], e_val[8];
    for (uint j = 0u; j < 8u; ++j) {
        r_hat[j] = 0.0f;
        ell[j] = 0.0f;
        e_val[j] = 0.0f;
    }

    const uint warm_off = tile_off > 20u ? (tile_off - 20u) : 0u;
    const uint c0_base = plug0[warm_off];
    const uint g_start = tile_off + t_start;
    const uint g_warm = g_start > 20u ? (g_start - 20u) : 0u;
    uint cur_e_idx = 0xffffffffu;
    for (uint g = g_warm; g < g_start; ++g) {
        const uint b = tokens[g] & 255u;
        const float alpha = gates_table[b * 2u];
        const float beta = gates_table[b * 2u + 1u];
        const float om_a = 1.0f - alpha;
        const float om_b = 1.0f - beta;
        const uint e_idx = plug0[g] - c0_base;
        if (e_idx != cur_e_idx) {
            cur_e_idx = e_idx;
            const uint e_base = e_idx * 256u + lane;
            for (uint j = 0u; j < 8u; ++j) {
                e_val[j] = tile_ema0_logits[e_base + 32u * j];
            }
        }
        const uint r_base = b * 256u + lane;
        for (uint j = 0u; j < 8u; ++j) {
            const float nr = fma(alpha, r_hat[j], om_a * r0_logits[r_base + 32u * j]);
            const float nl = fma(beta, ell[j], om_b * (nr + e_val[j]));
            r_hat[j] = nr;
            ell[j] = nl;
        }
    }

    for (uint t = t_start; t < t_end; ++t) {
        const uint g = tile_off + t;
        const uint b = tokens[g] & 255u;
        const float alpha = gates_table[b * 2u];
        const float beta = gates_table[b * 2u + 1u];
        const float om_a = 1.0f - alpha;
        const float om_b = 1.0f - beta;
        const uint e_idx = plug0[g] - c0_base;
        if (e_idx != cur_e_idx) {
            cur_e_idx = e_idx;
            const uint e_base = e_idx * 256u + lane;
            for (uint j = 0u; j < 8u; ++j) {
                e_val[j] = tile_ema0_logits[e_base + 32u * j];
            }
        }
        const uint r_base = b * 256u + lane;

        float local_top = -INFINITY;
        uint local_best = 256u;
        for (uint j = 0u; j < 8u; ++j) {
            const float nr = fma(alpha, r_hat[j], om_a * r0_logits[r_base + 32u * j]);
            const float nl = fma(beta, ell[j], om_b * (nr + e_val[j]));
            r_hat[j] = nr;
            ell[j] = nl;
            if (nl > local_top) {
                local_top = nl;
                local_best = lane + 32u * j;
            }
        }
        const float top = simd_max(local_top);
        const uint best = simd_min(local_top == top ? local_best : 256u);
        float total = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            total += exp(ell[j] - top);
        }
        total = simd_sum(total);
        const uint target_b = labels[g] & 255u;
        if (lane == 0u) {
            predictions[g] = best;
        }
        if (lane == (target_b & 31u)) {
            losses[g] = shape.score == 0u ? 0.0f : (top + log(total) - ell[target_b >> 5u]);
        }
    }
}

kernel void hnet2_readout_advance(
    device const float* ro_blk_abc [[buffer(0)]],
    device float* ro_blk_uv [[buffer(1)]],
    device float* r_logit_state [[buffer(2)]],
    device float* y0_logit_state [[buffer(3)]],
    constant HNetShape& shape [[buffer(4)]],
    uint v [[thread_index_in_threadgroup]]) {
    if (v >= 256u) return;
    const uint n_blk = min((shape.seq_len + 127u) / 128u, 256u);
    float r_hat = r_logit_state[v], ell = y0_logit_state[v];
    for (uint p = 0u; p < n_blk; ++p) {
        const float ap = ro_blk_abc[p * 3u + 0u];
        const float bp = ro_blk_abc[p * 3u + 1u];
        const float cp = ro_blk_abc[p * 3u + 2u];
        const uint u_idx = (p * 2u + 0u) * 256u + v;
        const uint v_idx = (p * 2u + 1u) * 256u + v;
        const float up = ro_blk_uv[u_idx];
        const float vp = ro_blk_uv[v_idx];
        ro_blk_uv[u_idx] = r_hat;
        ro_blk_uv[v_idx] = ell;
        const float next_r = fma(ap, r_hat, up);
        const float next_l = fma(cp, r_hat, fma(bp, ell, vp));
        r_hat = next_r;
        ell = next_l;
    }
    r_logit_state[v] = r_hat;
    y0_logit_state[v] = ell;
}
