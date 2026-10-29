#include <metal_stdlib>
using namespace metal;

constant uint kGradHeads = 8;
constant uint kGradDim = 64;
constant uint kGradQuery = kGradHeads * kGradDim;
constant uint kGradBlock = 64;
constant uint kGradSelected = 8;
constant uint kGradEntries = kGradBlock * kGradSelected;

struct PisaGradShape {
    uint start;
    uint rows;
    uint context;
    uint query_stride;
    uint kv_stride;
};

inline void add_float(device atomic_uint* address, float value) {
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address,
        &expected,
        as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed,
        memory_order_relaxed)) {}
}

// PISA holds the selected leaf indices fixed during backpropagation. This
// kernel recomputes the selected softmax and applies its exact adjoint. The
// query gradient is private to one row/head; key and value gradients are
// atomic because many causal queries may select the same source token.
kernel void fbt_pisa1_grad(
    device const half* query [[buffer(0)]],
    device const half* kv [[buffer(1)]],
    device const uint* blocks [[buffer(2)]],
    device const float* output_grad [[buffer(3)]],
    device atomic_uint* query_grad [[buffer(4)]],
    device atomic_uint* kv_grad [[buffer(5)]],
    constant PisaGradShape& p [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint local_row = group / kGradHeads;
    const uint head = group % kGradHeads;
    if (local_row >= p.rows) return;

    const uint row = p.start + local_row;
    const uint sequence = row / p.context;
    const uint position = row % p.context;
    const ulong q_base = ulong(local_row) * p.query_stride + head * kGradDim;
    const ulong dy_base = ulong(local_row) * kGradQuery + head * kGradDim;

    threadgroup float scores[kGradEntries];
    threadgroup float probabilities[kGradEntries];
    threadgroup float value_dots[kGradEntries];
    threadgroup float partial[2];
    threadgroup float shared;

    float local_max = -INFINITY;
    for (uint entry = tid; entry < kGradEntries; entry += 64) {
        const uint block = blocks[ulong(local_row) * kGradSelected + entry / kGradBlock];
        const uint token = entry % kGradBlock;
        const uint source_position = block == UINT_MAX ? UINT_MAX : block * kGradBlock + token;
        float score = -INFINITY;
        if (source_position <= position) {
            const uint source = sequence * p.context + source_position;
            const ulong k_base = ulong(source) * p.kv_stride;
            score = 0.0f;
            for (uint dim = 0; dim < kGradDim; ++dim)
                score = fma(float(query[q_base + dim]), float(kv[k_base + dim]), score);
            score *= 0.125f;
        }
        scores[entry] = score;
        local_max = max(local_max, score);
    }
    local_max = simd_max(local_max);
    if (lane == 0) partial[simd] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float value = lane < 2 ? partial[lane] : -INFINITY;
        const float maximum = simd_max(value);
        if (lane == 0) shared = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float maximum = shared;
    float local_total = 0.0f;
    for (uint entry = tid; entry < kGradEntries; entry += 64) {
        const float probability = isfinite(scores[entry]) ? exp(scores[entry] - maximum) : 0.0f;
        probabilities[entry] = probability;
        local_total += probability;
    }
    local_total = simd_sum(local_total);
    if (lane == 0) partial[simd] = local_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float value = lane < 2 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) shared = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_total = 1.0f / max(shared, 1.0e-20f);
    float local_center = 0.0f;
    for (uint entry = tid; entry < kGradEntries; entry += 64) {
        const uint block = blocks[ulong(local_row) * kGradSelected + entry / kGradBlock];
        const uint token = entry % kGradBlock;
        const uint source_position = block == UINT_MAX ? UINT_MAX : block * kGradBlock + token;
        const float probability = probabilities[entry] * inverse_total;
        probabilities[entry] = probability;
        float dot = 0.0f;
        if (source_position <= position) {
            const uint source = sequence * p.context + source_position;
            const ulong v_base = ulong(source) * p.kv_stride + kGradDim;
            for (uint dim = 0; dim < kGradDim; ++dim)
                dot = fma(output_grad[dy_base + dim], float(kv[v_base + dim]), dot);
        }
        value_dots[entry] = dot;
        local_center = fma(probability, dot, local_center);
    }
    local_center = simd_sum(local_center);
    if (lane == 0) partial[simd] = local_center;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float value = lane < 2 ? partial[lane] : 0.0f;
        const float center = simd_sum(value);
        if (lane == 0) shared = center;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float center = shared;
    if (tid < kGradDim) {
        float dq = 0.0f;
        for (uint entry = 0; entry < kGradEntries; ++entry) {
            const uint block = blocks[ulong(local_row) * kGradSelected + entry / kGradBlock];
            const uint token = entry % kGradBlock;
            const uint source_position = block == UINT_MAX ? UINT_MAX : block * kGradBlock + token;
            if (source_position <= position) {
                const uint source = sequence * p.context + source_position;
                const float ds = probabilities[entry] * (value_dots[entry] - center);
                dq = fma(0.125f * ds, float(kv[ulong(source) * p.kv_stride + tid]), dq);
            }
        }
        add_float(query_grad + q_base + tid, dq);
    }

    for (uint item = tid; item < kGradEntries * kGradDim; item += 64) {
        const uint entry = item / kGradDim;
        const uint dim = item % kGradDim;
        const uint block = blocks[ulong(local_row) * kGradSelected + entry / kGradBlock];
        const uint token = entry % kGradBlock;
        const uint source_position = block == UINT_MAX ? UINT_MAX : block * kGradBlock + token;
        if (source_position <= position) {
            const uint source = sequence * p.context + source_position;
            const float probability = probabilities[entry];
            const float ds = probability * (value_dots[entry] - center);
            const ulong base = ulong(source) * p.kv_stride;
            add_float(
                kv_grad + base + dim,
                0.125f * ds * float(query[q_base + dim]));
            add_float(
                kv_grad + base + kGradDim + dim,
                probability * output_grad[dy_base + dim]);
        }
    }
}
