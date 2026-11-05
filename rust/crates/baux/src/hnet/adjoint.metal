// Reverse the EMA and the selected-probability STE. Each channel owns its
// entire scan, so the probability contributions require no floating atomics.
kernel void hnet_dechunk_grad(
    device const float* streams [[buffer(0)]],
    device const half* core [[buffer(1)]],
    device const float* ema [[buffer(2)]],
    device const float* probs [[buffer(3)]],
    device const uint* plug [[buffer(4)]],
    device const uint* origins [[buffer(5)]],
    device const uint* count [[buffer(6)]],
    device float* combined_grad [[buffer(7)]],
    device float* prob_parts [[buffer(8)]],
    device float* core_grad [[buffer(9)]],
    constant HNetShape& shape [[buffer(10)]],
    uint d [[thread_position_in_grid]]) {
    const uint D = shape.d_model, L = shape.seq_len;
    if (d >= D) return;
    for (uint t = 0; t < L; ++t) {
        float g = 0.0f;
        for (uint s = 0; s < 4; ++s) g += streams[(t * 4u + s) * D + d];
        combined_grad[t * D + d] = g;
        prob_parts[t * D + d] = (probs[t] > 0.5f ? 1.0f : -1.0f) * g * ema[plug[t] * D + d];
        core_grad[t * D + d] = 0.0f;
    }
    uint t = L;
    float adjoint = 0.0f;
    for (uint c = count[0]; c-- > 0;) {
        while (t > 0 && plug[t - 1u] == c) {
            --t;
            adjoint += combined_grad[t * D + d];
        }
        const uint origin = origins[c];
        const float raw = probs[origin], p = clamp(raw, 1e-4f, 1.0f - 1e-4f);
        core_grad[c * D + d] = p * adjoint;
        const float previous = c > 0 ? ema[(c - 1u) * D + d] : 0.0f;
        if (raw >= 1e-4f && raw <= 1.0f - 1e-4f)
            prob_parts[origin * D + d] += adjoint * (float(core[c * D + d]) - previous);
        adjoint *= 1.0f - p;
    }
}

kernel void hnet_route_grad(
    device const float* projected [[buffer(0)]],
    device const float* prob_parts [[buffer(1)]],
    device float* projected_grad [[buffer(2)]],
    constant HNetShape& shape [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint D = shape.d_model, L = shape.seq_len;
    if (row == 0) {
        for (uint d = lane; d < D; d += 32u) {
            projected_grad[D + d] = 0.0f;
            projected_grad[(L - 1u) * 2u * D + d] = 0.0f;
        }
        return;
    }
    float nq = 0.0f, nk = 0.0f, dot = 0.0f, dp = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        const float q = projected[(row - 1u) * 2u * D + d];
        const float k = projected[row * 2u * D + D + d];
        nq = fma(q, q, nq); nk = fma(k, k, nk); dot = fma(q, k, dot);
        dp += prob_parts[row * D + d];
    }
    nq = sqrt(simd_sum(nq)); nk = sqrt(simd_sum(nk));
    const float qden = max(nq, 1e-12f), kden = max(nk, 1e-12f);
    const float cosine = simd_sum(dot) / (qden * kden);
    const float derivative = (cosine >= -1.0f && cosine <= 1.0f) ? -0.5f * simd_sum(dp) : 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        const float q = projected[(row - 1u) * 2u * D + d] / qden;
        const float k = projected[row * 2u * D + D + d] / kden;
        projected_grad[(row - 1u) * 2u * D + d] = derivative * (k - (nq >= 1e-12f ? cosine * q : 0.0f)) / qden;
        projected_grad[row * 2u * D + D + d] = derivative * (q - (nk >= 1e-12f ? cosine * k : 0.0f)) / kden;
    }
}

kernel void hnet_input_grad(
    device const half* weights [[buffer(0)]],
    device const float* projected_grad [[buffer(1)]],
    device const float* combined_grad [[buffer(2)]],
    device const float* chunk_stream_grad [[buffer(3)]],
    device const float* probs [[buffer(4)]],
    device const uint* plug [[buffer(5)]],
    device float* encoder_grad [[buffer(6)]],
    constant HNetShape& shape [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model;
    if (index >= shape.seq_len * D) return;
    const uint t = index / D, d = index % D;
    float result = 0.0f;
    if (t == 0 || probs[t] > 0.5f)
        for (uint s = 0; s < 4; ++s) result += chunk_stream_grad[(plug[t] * 4u + s) * D + d];
    for (uint j = 0; j < D; ++j) {
        result = fma(float(weights[j * D + d]), projected_grad[t * 2u * D + j], result);
        result = fma(float(weights[D * D + j * D + d]), projected_grad[t * 2u * D + D + j], result);
        result = fma(float(weights[2u * D * D + j * D + d]), combined_grad[t * D + j], result);
    }
    encoder_grad[index] = result;
}

kernel void hnet_weight_grad(
    device const half* encoder [[buffer(0)]],
    device const float* projected_grad [[buffer(1)]],
    device const float* combined_grad [[buffer(2)]],
    device float* weight_grad [[buffer(3)]],
    constant HNetShape& shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint D = shape.d_model;
    if (index >= 3u * D * D + D) return;
    const uint family = index / (D * D), offset = index % (D * D);
    const uint out = family < 3 ? offset / D : offset, in = offset % D;
    float result = 0.0f;
    for (uint t = 0; t < shape.seq_len; ++t) {
        const float g = family < 2 ? projected_grad[t * 2u * D + family * D + out] : combined_grad[t * D + out];
        result = fma(g, family == 3 ? 1.0f : float(encoder[t * D + in]), result);
    }
    weight_grad[index] += result;
}
