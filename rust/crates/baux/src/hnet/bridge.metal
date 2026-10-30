// The hierarchy boundary uses normalized D-wide states; the PISA/mHC core
// receives four replicated streams. Padding is after all real chunks.
kernel void hnet_mhc_project(
    device const half* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device float* projected [[buffer(2)]],
    device float* residual [[buffer(3)]],
    constant HNetShape& shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model;
    if (index >= shape.seq_len * D) return;
    const uint row = index / D, col = index % D;
    float q = 0.0f, k = 0.0f, r = float(weights[3u * D * D + col]);
    for (uint j = 0; j < D; ++j) {
        const float x = float(input[row * D + j]);
        q = fma(float(weights[col * D + j]), x, q);
        k = fma(float(weights[D * D + col * D + j]), x, k);
        r = fma(float(weights[2u * D * D + col * D + j]), x, r);
    }
    projected[row * 2u * D + col] = q;
    projected[row * 2u * D + D + col] = k;
    residual[index] = r;
}

kernel void hnet_mhc_route(
    device const float* projected [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant HNetShape& shape [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint D = shape.d_model;
    float dot = 0.0f, nq = 0.0f, nk = 0.0f;
    if (row > 0) {
        for (uint d = lane; d < D; d += 32u) {
            const float q = projected[(row - 1u) * 2u * D + d];
            const float k = projected[row * 2u * D + D + d];
            dot = fma(q, k, dot);
            nq = fma(q, q, nq);
            nk = fma(k, k, nk);
        }
    }
    const float cosine = simd_sum(dot) / (max(sqrt(simd_sum(nq)), 1e-12f) * max(sqrt(simd_sum(nk)), 1e-12f));
    if (lane == 0) probs[row] = row == 0 ? 1.0f : clamp((1.0f - cosine) * 0.5f, 0.0f, 1.0f);
}

kernel void hnet_mhc_route_state(
    device const float* projected [[buffer(0)]],
    device float* probs [[buffer(1)]],
    device const float* q_state [[buffer(2)]],
    device const uint* has_seen [[buffer(3)]],
    constant HNetShape& shape [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint D = shape.d_model;
    const bool first_token = (row == 0u) && (has_seen[0] == 0u || shape.score == 2u);
    float dot = 0.0f, nq = 0.0f, nk = 0.0f;
    if (!first_token) {
        for (uint d = lane; d < D; d += 32u) {
            const float q = row == 0u ? q_state[d] : projected[(row - 1u) * 2u * D + d];
            const float k = projected[row * 2u * D + D + d];
            dot = fma(q, k, dot);
            nq = fma(q, q, nq);
            nk = fma(k, k, nk);
        }
    }
    const float cosine = simd_sum(dot) / (max(sqrt(simd_sum(nq)), 1e-12f) * max(sqrt(simd_sum(nk)), 1e-12f));
    if (lane == 0) probs[row] = first_token ? 1.0f : clamp((1.0f - cosine) * 0.5f, 0.0f, 1.0f);
}

kernel void hnet_mhc_chunk(
    device const half* input [[buffer(0)]],
    device half* streams [[buffer(1)]],
    device half* encoder_input [[buffer(2)]],
    device uint* origins [[buffer(3)]],
    device const float* probs [[buffer(4)]],
    device uint* plug [[buffer(5)]],
    device uint* count [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint L = shape.seq_len, D = shape.d_model;
    threadgroup uint sg_totals[8];
    uint running_base = 0u;
    for (uint base = 0u; base < L; base += 256u) {
        const uint t = base + tid;
        const uint is_b = (t < L && (t == 0u || probs[t] > 0.5f)) ? 1u : 0u;
        const uint sg_prefix = simd_prefix_exclusive_sum(is_b);
        const uint sg_sum = simd_sum(is_b);
        if (lane == 0u) sg_totals[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg_offset = 0u, block_total = 0u;
        for (uint s = 0u; s < 8u; ++s) {
            if (s < sg_id) sg_offset += sg_totals[s];
            block_total += sg_totals[s];
        }
        const uint chunk_idx = running_base + sg_offset + sg_prefix;
        if (t < L) {
            if (is_b != 0u) origins[chunk_idx] = t;
            plug[t] = (chunk_idx + is_b) - 1u;
        }
        running_base += block_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) count[0] = running_base;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < L * D; i += 256u) {
        const uint row = i / D, col = i % D;
        encoder_input[i] = input[i];
        const half val = row < running_base ? input[origins[row] * D + col] : half(0.0f);
        for (uint s = 0u; s < 4u; ++s) streams[(row * 4u + s) * D + col] = val;
    }
}

kernel void hnet_mhc_chunk_scan(
    device uint* origins [[buffer(0)]],
    device const float* probs [[buffer(1)]],
    device uint* plug [[buffer(2)]],
    device uint* count [[buffer(3)]],
    device const float* projected [[buffer(4)]],
    device float* q_state [[buffer(5)]],
    device const uint* has_seen [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint L = shape.seq_len, D = shape.d_model;
    const bool first_is_boundary = (has_seen[0] == 0u);
    threadgroup uint sg_totals[8];
    uint running_base = 0u;
    for (uint base = 0u; base < L; base += 256u) {
        const uint t = base + tid;
        const uint is_b = (t < L && ((t == 0u && first_is_boundary) || probs[t] > 0.5f)) ? 1u : 0u;
        const uint sg_prefix = simd_prefix_exclusive_sum(is_b);
        const uint sg_sum = simd_sum(is_b);
        if (lane == 0u) sg_totals[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg_offset = 0u, block_total = 0u;
        for (uint s = 0u; s < 8u; ++s) {
            if (s < sg_id) sg_offset += sg_totals[s];
            block_total += sg_totals[s];
        }
        const uint chunk_idx = running_base + sg_offset + sg_prefix;
        if (t < L) {
            if (is_b != 0u) origins[chunk_idx] = t;
            const uint seen = chunk_idx + is_b;
            plug[t] = seen > 0u ? (seen - 1u) : 0xFFFFFFFFu;
        }
        running_base += block_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (L > 0u) {
        for (uint d = tid; d < D; d += 256u) {
            q_state[d] = projected[(L - 1u) * 2u * D + d];
        }
    }
    if (tid == 0u) {
        count[0] = running_base;
    }
}

kernel void hnet_mhc_chunk_pack(
    device const half* input [[buffer(0)]],
    device half* streams [[buffer(1)]],
    device half* encoder_input [[buffer(2)]],
    device const uint* origins [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    constant HNetShape& shape [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model, L = shape.seq_len, M = max(L, shape.max_chunks);
    if (index >= M * D) return;
    const uint row = index / D, col = index % D;
    if (row < L) encoder_input[index] = input[index];
    const uint active = count[0];
    if (row < active) {
        const half val = input[origins[row] * D + col];
        for (uint s = 0u; s < 4u; ++s) streams[(row * 4u + s) * D + col] = val;
    } else if (row < shape.max_chunks) {
        for (uint s = 0u; s < 4u; ++s) streams[(row * 4u + s) * D + col] = half(0.0f);
    }
}

constant uint HNET_EMA_PARTS = 128u;

kernel void hnet_mhc_dechunk_reduce(
    device const half* core [[buffer(0)]],
    device float* ema [[buffer(1)]],
    device const uint* origins [[buffer(2)]],
    device const float* probs [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    device half* saved_core [[buffer(5)]],
    device float* part_a [[buffer(6)]],
    device float* part_b [[buffer(7)]],
    device const float* ema_in [[buffer(8)]],
    constant HNetShape& shape [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, part = gid.y;
    const uint D = shape.d_model;
    if (d >= D || part >= HNET_EMA_PARTS) return;
    const uint total = count[0];
    const uint part_size = max((total + HNET_EMA_PARTS - 1u) / HNET_EMA_PARTS, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    if (c_start >= c_end) {
        part_a[part * D + d] = 1.0f;
        part_b[part * D + d] = 0.0f;
        return;
    }
    if (part == 0u) {
        float state = ema_in[d];
        for (uint c = c_start; c < c_end; ++c) {
            const half x = core[c * D + d];
            saved_core[c * D + d] = x;
            const float p = clamp(probs[origins[c]], 1e-4f, 1.0f - 1e-4f);
            state = fma(p, float(x), (1.0f - p) * state);
            ema[c * D + d] = state;
        }
        part_a[d] = 0.0f;
        part_b[d] = state;
    } else {
        float a = 1.0f, b = 0.0f;
        for (uint c = c_start; c < c_end; ++c) {
            const half x = core[c * D + d];
            saved_core[c * D + d] = x;
            const float p = clamp(probs[origins[c]], 1e-4f, 1.0f - 1e-4f);
            a = (1.0f - p) * a;
            b = fma(p, float(x), (1.0f - p) * b);
        }
        part_a[part * D + d] = a;
        part_b[part * D + d] = b;
    }
}

kernel void hnet_mhc_dechunk_scan(
    device const half* core [[buffer(0)]],
    device float* ema [[buffer(1)]],
    device const uint* origins [[buffer(2)]],
    device const float* probs [[buffer(3)]],
    device const uint* count [[buffer(4)]],
    device const float* part_a [[buffer(5)]],
    device const float* part_b [[buffer(6)]],
    device const float* ema_in [[buffer(7)]],
    device float* ema_out [[buffer(8)]],
    constant HNetShape& shape [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, part = gid.y;
    const uint D = shape.d_model;
    if (d >= D || part >= HNET_EMA_PARTS) return;
    const uint total = count[0];
    const uint part_size = max((total + HNET_EMA_PARTS - 1u) / HNET_EMA_PARTS, 1u);
    const uint c_start = min(part * part_size, total);
    const uint c_end = min(c_start + part_size, total);
    if (part == 0u) {
        if (total == 0u) {
            ema_out[d] = ema_in[d];
        } else if (c_end == total) {
            ema_out[d] = part_b[d];
        }
        return;
    }
    if (c_start >= c_end) return;
    float state = ema_in[d];
    for (uint k = 0u; k < part; ++k) {
        state = fma(part_a[k * D + d], state, part_b[k * D + d]);
    }
    for (uint c = c_start; c < c_end; ++c) {
        const float p = clamp(probs[origins[c]], 1e-4f, 1.0f - 1e-4f);
        state = fma(p, float(core[c * D + d]), (1.0f - p) * state);
        ema[c * D + d] = state;
    }
    if (c_end == total) {
        ema_out[d] = state;
    }
}

kernel void hnet_mhc_dechunk_plug(
    device half* streams [[buffer(0)]],
    device const float* ema [[buffer(1)]],
    device const uint* plug [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    device const float* residual [[buffer(4)]],
    device const float* ema_in [[buffer(5)]],
    constant HNetShape& shape [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model, L = shape.seq_len;
    if (index >= L * D) return;
    const uint t = index / D, d = index % D;
    const uint p_idx = plug[t];
    const float e = p_idx < count[0] ? ema[p_idx * D + d] : ema_in[d];
    const half value = half(e + residual[index]);
    for (uint s = 0u; s < 4u; ++s) streams[(t * 4u + s) * D + d] = value;
}

kernel void hnet_mhc_dechunk(
    device const half* core [[buffer(0)]],
    device half* streams [[buffer(1)]],
    device float* ema [[buffer(2)]],
    device const uint* origins [[buffer(3)]],
    device const float* probs [[buffer(4)]],
    device const uint* plug [[buffer(5)]],
    device const uint* count [[buffer(6)]],
    device const float* residual [[buffer(7)]],
    device half* saved_core [[buffer(8)]],
    constant HNetShape& shape [[buffer(9)]],
    uint d [[thread_position_in_grid]]) {
    const uint D = shape.d_model, L = shape.seq_len;
    if (d >= D) return;
    float state = 0.0f;
    for (uint c = 0; c < count[0]; ++c) {
        saved_core[c * D + d] = core[c * D + d];
        const float p = clamp(probs[origins[c]], 1e-4f, 1.0f - 1e-4f);
        state = fma(p, float(core[c * D + d]), (1.0f - p) * state);
        ema[c * D + d] = state;
    }
    for (uint t = 0; t < L; ++t) {
        const half value = half(ema[plug[t] * D + d] + residual[t * D + d]);
        for (uint s = 0; s < 4; ++s) streams[(t * 4u + s) * D + d] = value;
    }
}

// Exact flat byte softmax. The tied embedding is stored [D, 256].
kernel void hnet_mhc_readout(
    device const half* hidden [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const uint* labels [[buffer(2)]],
    device uint* predictions [[buffer(3)]],
    device float* losses [[buffer(4)]],
    constant HNetShape& shape [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float logits[8], top = -INFINITY;
    for (uint j = 0; j < 8; ++j) {
        const uint byte = lane + 32u * j;
        float value = 0.0f;
        for (uint d = 0; d < shape.d_model; ++d)
            value = fma(float(hidden[row * shape.d_model + d]), float(weight[d * 256u + byte]), value);
        logits[j] = value;
        top = max(top, value);
    }
    top = simd_max(top);
    float total = 0.0f, target = 0.0f;
    uint best = 256;
    for (uint j = 0; j < 8; ++j) {
        const uint byte = lane + 32u * j;
        total += exp(logits[j] - top);
        if (byte == labels[row]) target = logits[j];
        if (logits[j] == top) best = min(best, byte);
    }
    total = simd_sum(total);
    target = simd_sum(target);
    best = simd_min(best);
    if (lane == 0) {
        predictions[row] = best;
        losses[row] = shape.score == 0 ? 0.0f : top + log(total) - target;
    }
}

kernel void hnet_mhc_logits(
    device const half* hidden [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint byte [[thread_position_in_grid]]) {
    if (byte >= 256u) return;
    float value = 0.0f;
    for (uint d = 0; d < width; ++d)
        value = fma(float(hidden[d]), float(weight[d * 256u + byte]), value);
    logits[byte] = value;
}
