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

kernel void hnet_mhc_chunk(
    device const half* input [[buffer(0)]],
    device half* streams [[buffer(1)]],
    device half* encoder_input [[buffer(2)]],
    device uint* origins [[buffer(3)]],
    device const float* probs [[buffer(4)]],
    device uint* plug [[buffer(5)]],
    device uint* count [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint L = shape.seq_len, D = shape.d_model;
    if (tid == 0) {
        uint chunks = 0;
        for (uint t = 0; t < L; ++t) {
            if (t == 0 || probs[t] > 0.5f) origins[chunks++] = t;
            plug[t] = chunks - 1u;
        }
        count[0] = chunks;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < L * D; i += 256u) encoder_input[i] = input[i];
    for (uint i = tid; i < L * 4u * D; i += 256u) {
        const uint row = i / (4u * D), col = i % D;
        streams[i] = row < count[0] ? input[origins[row] * D + col] : half(0.0f);
    }
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
