#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong elements;
    ulong stride;
};

struct AuditSpan {
    ulong offset;
    uint elements;
    uint family;
};

kernel void audit_gradients(
    device const float* values [[buffer(0)]],
    device const AuditSpan* spans [[buffer(1)]],
    device float2* output [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    AuditSpan span = spans[group];
    float invalid = 0.0f;
    float maximum = 0.0f;
    for (uint i = tid; i < span.elements; i += 256) {
        float value = values[span.offset + i];
        invalid += isfinite(value) ? 0.0f : 1.0f;
        if (isfinite(value)) maximum = max(maximum, abs(value));
    }
    invalid = simd_sum(invalid);
    maximum = simd_max(maximum);
    threadgroup float2 partial[8];
    if (lane == 0) partial[simd] = float2(invalid, maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float count = simd_sum(lane < 8 ? partial[lane].x : 0.0f);
        float peak = simd_max(lane < 8 ? partial[lane].y : 0.0f);
        if (lane == 0) output[group] = float2(count, peak);
    }
}

kernel void audit_weights(
    device const half* values [[buffer(0)]],
    device const AuditSpan* spans [[buffer(1)]],
    device float2* output [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    AuditSpan span = spans[group];
    float invalid = 0.0f;
    float maximum = 0.0f;
    for (uint i = tid; i < span.elements; i += 256) {
        float value = float(values[span.offset + i]);
        invalid += isfinite(value) ? 0.0f : 1.0f;
        if (isfinite(value)) maximum = max(maximum, abs(value));
    }
    invalid = simd_sum(invalid);
    maximum = simd_max(maximum);
    threadgroup float2 partial[8];
    if (lane == 0) partial[simd] = float2(invalid, maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float count = simd_sum(lane < 8 ? partial[lane].x : 0.0f);
        float peak = simd_max(lane < 8 ? partial[lane].y : 0.0f);
        if (lane == 0) output[group] = float2(count, peak);
    }
}

inline float decode_weight(ushort value) {
    return float(as_type<half>(value));
}

kernel void update_alignment(
    device const ushort* incumbent [[buffer(0)]],
    device const ushort* candidate [[buffer(1)]],
    device const ushort* muon [[buffer(2)]],
    device float4* partials [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    float4 moment = float4(0.0f);
    float3 invalid = float3(0.0f);
    for (ulong i = index; i < p.elements; i += p.stride) {
        float base = decode_weight(incumbent[i]);
        float candidate_value = decode_weight(candidate[i]);
        float muon_value = decode_weight(muon[i]);
        float left = candidate_value - base;
        float right = muon_value - base;
        bool base_ok = isfinite(base);
        bool candidate_ok = isfinite(candidate_value);
        bool muon_ok = isfinite(muon_value);
        if (base_ok && candidate_ok && muon_ok) {
            moment.xyz += float3(left * right, left * left, right * right);
        }
        invalid += float3(
            base_ok ? 0.0f : 1.0f,
            candidate_ok ? 0.0f : 1.0f,
            muon_ok ? 0.0f : 1.0f);
    }
    moment = simd_sum(moment);
    invalid = simd_sum(invalid);
    threadgroup float4 sums[8];
    threadgroup float3 invalids[8];
    if (lane == 0) sums[simd] = moment;
    if (lane == 0) invalids[simd] = invalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float4 value = simd_sum(lane < 8 ? sums[lane] : float4(0.0f));
        float3 bad = simd_sum(lane < 8 ? invalids[lane] : float3(0.0f));
        if (lane == 0) {
            partials[2 * group] = value;
            partials[2 * group + 1] = float4(bad, 0.0f);
        }
    }
}
