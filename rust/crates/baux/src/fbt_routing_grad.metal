#include <metal_stdlib>
using namespace metal;

struct Top3RouteShape {
    uint rows;
    uint width;
    uint experts;
    uint top_k;
    uint block_tokens;
    uint blocks;
};

constant uint kExpertWidth = 216;
constant uint kGateWidth = 432;
constant uint kActivationStride = 224;

inline void moe_grad_add(device atomic_uint* address, float value) {
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &expected, as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed, memory_order_relaxed)) {}
}

inline float sigmoid_stable(float value) {
    const float e = exp(-abs(value));
    return value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// Reverse the shared-plus-top-3 gather. Each packed row belongs to exactly one
// assignment, so routed gradients do not require atomic accumulation.
kernel void moe_combine_grad(
    device const half* route_weights [[buffer(0)]],
    device const uint* packed_rows [[buffer(1)]],
    device const half* routed_output [[buffer(2)]],
    device const float* branch_grad [[buffer(3)]],
    device float* routed_grad [[buffer(4)]],
    device float* shared_grad [[buffer(5)]],
    device float* route_grad [[buffer(6)]],
    constant Top3RouteShape& p [[buffer(7)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (token >= p.rows) return;
    threadgroup uint packed[3];
    threadgroup float weight[3];
    threadgroup float partial[3][4];
    if (tid < p.top_k) {
        const ulong assignment = ulong(token) * p.top_k + tid;
        packed[tid] = packed_rows[assignment];
        weight[tid] = float(route_weights[assignment]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local[3] = {0.0f, 0.0f, 0.0f};
    for (uint column = tid; column < p.width; column += 128) {
        const ulong output = ulong(token) * p.width + column;
        const float dy = branch_grad[output];
        shared_grad[output] = dy;
        for (uint route = 0; route < p.top_k; ++route) {
            const ulong source = ulong(packed[route]) * p.width + column;
            routed_grad[source] = weight[route] * dy;
            local[route] = fma(dy, float(routed_output[source]), local[route]);
        }
    }
    for (uint route = 0; route < p.top_k; ++route) {
        const float value = simd_sum(local[route]);
        if (lane == 0) partial[route][simd] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        for (uint route = 0; route < p.top_k; ++route) {
            const float value = simd_sum(lane < 4 ? partial[route][lane] : 0.0f);
            if (lane == 0)
                route_grad[ulong(token) * p.top_k + route] = value;
        }
    }
}

kernel void moe_routed_down_input_grad(
    device const half* weights [[buffer(0)]],
    device const uint* packed_experts [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    device float* activation_grad [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint assignments = p.rows * p.top_k;
    if (gid >= assignments * kExpertWidth) return;
    const uint row = gid / kExpertWidth;
    const uint hidden = gid % kExpertWidth;
    const uint expert = packed_experts[row] + 1;
    const ulong weight = ulong(expert) * kExpertWidth * p.width + ulong(hidden) * p.width;
    const ulong dy = ulong(row) * p.width;
    float value = 0.0f;
    for (uint column = 0; column < p.width; ++column)
        value = fma(output_grad[dy + column], float(weights[weight + column]), value);
    activation_grad[ulong(row) * kActivationStride + hidden] = value;
}

kernel void moe_shared_down_input_grad(
    device const half* weights [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device float* activation_grad [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * kExpertWidth) return;
    const uint row = gid / kExpertWidth;
    const uint hidden = gid % kExpertWidth;
    float value = 0.0f;
    for (uint column = 0; column < p.width; ++column)
        value = fma(output_grad[ulong(row) * p.width + column],
                    float(weights[ulong(hidden) * p.width + column]), value);
    activation_grad[ulong(row) * kActivationStride + hidden] = value;
}

kernel void moe_routed_down_weight_grad(
    device const half* activation [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const uint* expert_offsets [[buffer(2)]],
    device const uint* expert_loads [[buffer(3)]],
    device float* weight_grad [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong per_expert = ulong(kExpertWidth) * p.width;
    const ulong total = ulong(p.experts) * per_expert;
    if (ulong(gid) >= total) return;
    const uint expert = gid / per_expert;
    const uint item = gid % per_expert;
    const uint hidden = item / p.width;
    const uint column = item % p.width;
    const uint first = expert_offsets[expert];
    const uint end = first + expert_loads[expert];
    float value = 0.0f;
    for (uint row = first; row < end; ++row)
        value = fma(float(activation[ulong(row) * kActivationStride + hidden]),
                    output_grad[ulong(row) * p.width + column], value);
    weight_grad[ulong(expert + 1) * per_expert + item] += value;
}

kernel void moe_shared_down_weight_grad(
    device const half* activation [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device float* weight_grad [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= kExpertWidth * p.width) return;
    const uint hidden = gid / p.width;
    const uint column = gid % p.width;
    float value = 0.0f;
    for (uint row = 0; row < p.rows; ++row)
        value = fma(float(activation[ulong(row) * kActivationStride + hidden]),
                    output_grad[ulong(row) * p.width + column], value);
    weight_grad[gid] += value;
}

kernel void moe_swiglu_grad(
    device const half* projected [[buffer(0)]],
    device const float* activation_grad [[buffer(1)]],
    device float* projection_grad [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * kExpertWidth) return;
    const uint row = gid / kExpertWidth;
    const uint column = gid % kExpertWidth;
    const ulong base = ulong(row) * kGateWidth;
    const float gate = float(projected[base + column]);
    const float up = float(projected[base + kExpertWidth + column]);
    const float sigmoid = sigmoid_stable(gate);
    const float dy = activation_grad[ulong(row) * kActivationStride + column];
    projection_grad[base + column] = dy * up *
        (sigmoid + gate * sigmoid * (1.0f - sigmoid));
    projection_grad[base + kExpertWidth + column] = dy * gate * sigmoid;
}

kernel void moe_routed_gate_input_grad(
    device const half* weights [[buffer(0)]],
    device const uint* packed_experts [[buffer(1)]],
    device const float* projection_grad [[buffer(2)]],
    device atomic_uint* packed_input_grad [[buffer(3)]],
    constant Top3RouteShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint assignments = p.rows * p.top_k;
    if (gid >= assignments * p.width) return;
    const uint row = gid / p.width;
    const uint input = gid % p.width;
    const uint expert = packed_experts[row] + 1;
    const ulong weight = ulong(expert) * p.width * kGateWidth + ulong(input) * kGateWidth;
    const ulong dy = ulong(row) * kGateWidth;
    float value = 0.0f;
    for (uint column = 0; column < kGateWidth; ++column)
        value = fma(projection_grad[dy + column], float(weights[weight + column]), value);
    moe_grad_add(packed_input_grad + ulong(row) * p.width + input, value);
}

kernel void moe_shared_gate_input_grad(
    device const half* weights [[buffer(0)]],
    device const float* projection_grad [[buffer(1)]],
    device atomic_uint* input_grad [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) return;
    const uint row = gid / p.width;
    const uint input = gid % p.width;
    float value = 0.0f;
    for (uint column = 0; column < kGateWidth; ++column)
        value = fma(projection_grad[ulong(row) * kGateWidth + column],
                    float(weights[ulong(input) * kGateWidth + column]), value);
    moe_grad_add(input_grad + gid, value);
}

kernel void moe_routed_gate_weight_grad(
    device const half* packed_input [[buffer(0)]],
    device const float* projection_grad [[buffer(1)]],
    device const uint* expert_offsets [[buffer(2)]],
    device const uint* expert_loads [[buffer(3)]],
    device float* weight_grad [[buffer(4)]],
    constant Top3RouteShape& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong per_expert = ulong(p.width) * kGateWidth;
    const ulong total = ulong(p.experts) * per_expert;
    if (ulong(gid) >= total) return;
    const uint expert = gid / per_expert;
    const uint item = gid % per_expert;
    const uint input = item / kGateWidth;
    const uint column = item % kGateWidth;
    const uint first = expert_offsets[expert];
    const uint end = first + expert_loads[expert];
    float value = 0.0f;
    for (uint row = first; row < end; ++row)
        value = fma(float(packed_input[ulong(row) * p.width + input]),
                    projection_grad[ulong(row) * kGateWidth + column], value);
    weight_grad[ulong(expert + 1) * per_expert + item] += value;
}

kernel void moe_shared_gate_weight_grad(
    device const half* input [[buffer(0)]],
    device const float* projection_grad [[buffer(1)]],
    device float* weight_grad [[buffer(2)]],
    constant Top3RouteShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.width * kGateWidth) return;
    const uint input_column = gid / kGateWidth;
    const uint output_column = gid % kGateWidth;
    float value = 0.0f;
    for (uint row = 0; row < p.rows; ++row)
        value = fma(float(input[ulong(row) * p.width + input_column]),
                    projection_grad[ulong(row) * kGateWidth + output_column], value);
    weight_grad[gid] += value;
}

// The selected expert set is held fixed. Differentiate normalized sigmoid
// weights and retain three selected-logit adjoints per token.
kernel void moe_router_score_grad(
    device const half* input [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device const uint* route_experts [[buffer(2)]],
    device const half* route_weights [[buffer(3)]],
    device const float* route_grad [[buffer(4)]],
    device float* score_grad [[buffer(5)]],
    constant Top3RouteShape& p [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (token >= p.rows) return;
    threadgroup float partial[3][4];
    threadgroup float logits[3];
    for (uint route = 0; route < p.top_k; ++route) {
        const ulong assignment = ulong(token) * p.top_k + route;
        const uint expert = route_experts[assignment];
        float local = 0.0f;
        for (uint column = tid; column < p.width; column += 128)
            local = fma(float(input[ulong(token) * p.width + column]),
                        float(router[ulong(column) * p.experts + expert]), local);
        local = simd_sum(local);
        if (lane == 0) partial[route][simd] = local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        for (uint route = 0; route < p.top_k; ++route) {
            const float value = simd_sum(lane < 4 ? partial[route][lane] : 0.0f);
            if (lane == 0) logits[route] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float center = 0.0f;
    for (uint route = 0; route < p.top_k; ++route) {
        const ulong assignment = ulong(token) * p.top_k + route;
        center = fma(route_grad[assignment], float(route_weights[assignment]), center);
    }
    if (tid < p.top_k) {
        const uint route = tid;
        const ulong assignment = ulong(token) * p.top_k + route;
        const float sigmoid = sigmoid_stable(logits[route]);
        const float denominator = sigmoid / max(float(route_weights[assignment]), 1.0e-20f);
        score_grad[assignment] = (route_grad[assignment] - center) *
            sigmoid * (1.0f - sigmoid) / max(denominator, 1.0e-20f);
    }
}

kernel void moe_router_project_grad(
    device const half* input [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device const uint* route_experts [[buffer(2)]],
    device const float* score_grad [[buffer(3)]],
    device const float* packed_input_grad [[buffer(4)]],
    device const uint* packed_rows [[buffer(5)]],
    device atomic_uint* input_grad [[buffer(6)]],
    device atomic_uint* router_grad [[buffer(7)]],
    constant Top3RouteShape& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    float dx = 0.0f;
    for (uint route = 0; route < p.top_k; ++route) {
        const ulong assignment = ulong(token) * p.top_k + route;
        const uint expert = route_experts[assignment];
        const float dlogit = score_grad[assignment];
        dx = fma(dlogit, float(router[ulong(column) * p.experts + expert]), dx);
        moe_grad_add(router_grad + ulong(column) * p.experts + expert,
                     dlogit * float(input[ulong(token) * p.width + column]));
        const uint packed = packed_rows[assignment];
        dx += packed_input_grad[ulong(packed) * p.width + column];
    }
    moe_grad_add(input_grad + gid, dx);
}
