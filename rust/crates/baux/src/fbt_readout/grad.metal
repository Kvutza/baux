// Identity adjoints at query/centroid half casts; all adjoints are for the
// mean target NLL. Parameter contributions accumulate into shared gradients.
// WaveShape and mhc4_offset come from the compiled production forward source.

METAL_FUNC void node_add(device atomic_uint* address, float value) {
    uint previous = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &previous, as_type<uint>(as_type<float>(previous) + value),
        memory_order_relaxed, memory_order_relaxed)) {}
}

kernel void tree_adjoint(
    device const half* hidden [[buffer(0)]],
    device const half* qkv [[buffer(1)]],
    device const half* readout [[buffer(2)]],
    device const half* centroids [[buffer(3)]],
    device const uint* targets [[buffer(4)]],
    device float* queries [[buffer(5)]],
    device float* errors [[buffer(6)]],
    device float* query_grad [[buffer(7)]],
    device atomic_uint* node_grad [[buffer(8)]],
    constant WaveShape& p [[buffer(9)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.tokens) return;
    float q[64];
    float dq[64];
    for (uint d = 0; d < 64; ++d) {
        float sum = 0.0f;
        for (uint h = 0; h < 512; ++h)
            sum = fma(float(hidden[row * 512 + h]), float(qkv[h * 640 + d]), sum);
        q[d] = float(half(sum * 2.82842712f));
        queries[row * 64 + d] = q[d];
        dq[d] = 0.0f;
    }
    for (uint level = 0; level < 6; ++level) {
        const uint parent = targets[row] >> ((6 - level) * 3);
        const uint target_child = (targets[row] >> ((5 - level) * 3)) & 7u;
        float logits[8];
        float peak = -INFINITY;
        for (uint child = 0; child < 8; ++child) {
            const uint index = parent * 8 + child;
            float sum = 0.0f;
            for (uint d = 0; d < 64; ++d) {
                const float weight = level < 5
                    ? float(centroids[(mhc4_offset(level) + index) * 64 + d])
                    : float(readout[d * p.vocabulary + index]);
                sum = fma(q[d], weight, sum);
            }
            logits[child] = sum;
            peak = max(peak, sum);
        }
        float denominator = 0.0f;
        for (uint child = 0; child < 8; ++child)
            denominator += exp(logits[child] - peak);
        for (uint child = 0; child < 8; ++child) {
            const uint index = parent * 8 + child;
            const float error = (exp(logits[child] - peak) / denominator
                - float(child == target_child)) / float(p.tokens);
            errors[row * 48 + level * 8 + child] = error;
            for (uint d = 0; d < 64; ++d) {
                const float weight = level < 5
                    ? float(centroids[(mhc4_offset(level) + index) * 64 + d])
                    : float(readout[d * p.vocabulary + index]);
                dq[d] = fma(error, weight, dq[d]);
                const uint node = level < 5 ? mhc4_offset(level) + index : 37448 + index;
                node_add(node_grad + node * 64 + d, error * q[d]);
            }
        }
    }
    for (uint d = 0; d < 64; ++d) query_grad[row * 64 + d] = dq[d];
}

kernel void leaf_adjoint(
    device const float* node_grad [[buffer(0)]],
    device float* gradient [[buffer(1)]],
    constant WaveShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 64 * p.vocabulary) return;
    const uint dim = gid / p.vocabulary;
    const uint token = gid % p.vocabulary;
    float sum = node_grad[(37448 + token) * 64 + dim];
    for (uint level = 0; level < 5; ++level) {
        const uint leaves = 1u << ((5 - level) * 3);
        const uint node = mhc4_offset(level) + token / leaves;
        sum += node_grad[node * 64 + dim] * rsqrt(float(leaves));
    }
    gradient[gid] += sum;
}

kernel void project_adjoint(
    device const half* hidden [[buffer(0)]],
    device const half* qkv [[buffer(1)]],
    device const float* query_grad [[buffer(2)]],
    device float* hidden_grad [[buffer(3)]],
    device float* projection_grad [[buffer(4)]],
    constant WaveShape& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < 512 * 640) {
        const uint h = gid / 640;
        const uint d = gid % 640;
        float sum = 0.0f;
        if (d < 64) {
            for (uint row = 0; row < p.tokens; ++row)
                sum += float(hidden[row * 512 + h]) * query_grad[row * 64 + d];
        }
        projection_grad[gid] += sum * 2.82842712f;
    }
    if (gid < p.tokens * 512) {
        const uint row = gid / 512;
        const uint h = gid % 512;
        float sum = 0.0f;
        for (uint d = 0; d < 64; ++d)
            sum += float(qkv[h * 640 + d]) * query_grad[row * 64 + d];
        hidden_grad[gid] += sum * 2.82842712f;
    }
}
