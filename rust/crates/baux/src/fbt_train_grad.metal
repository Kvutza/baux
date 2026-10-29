#include <metal_stdlib>
using namespace metal;

struct DenseShape { uint rows, input, output, reserved; };

kernel void gradient_half(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = half(input[gid]);
}

#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace mpp::tensor_ops;

kernel void dense_input_tensor(
    device half* weights [[buffer(0)]],
    device half* output_grad [[buffer(1)]],
    device float* input_grad [[buffer(2)]],
    constant DenseShape& p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        128, 64, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto dy = tensor(output_grad,
        dextents<int32_t, 2>{int32_t(p.output), int32_t(p.rows)});
    auto weight = tensor(weights,
        dextents<int32_t, 2>{int32_t(p.output), int32_t(p.input)});
    auto dx = tensor(input_grad,
        dextents<int32_t, 2>{int32_t(p.input), int32_t(p.rows)});
    auto dy_tile = dy.slice(0, tile.y * 128);
    auto weight_tile = weight.slice(0, tile.x * 64);
    auto dx_tile = dx.slice(tile.x * 64, tile.y * 128);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(dy_tile), decltype(weight_tile), float>();
    result.load(dx_tile);
    op.run(dy_tile, weight_tile, result);
    result.store(dx_tile);
}

kernel void dense_weight_tensor(
    device half* input [[buffer(0)]],
    device half* output_grad [[buffer(1)]],
    device float* weight_grad [[buffer(2)]],
    constant DenseShape& p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        64, 64, static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto x = tensor(input,
        dextents<int32_t, 2>{int32_t(p.input), int32_t(p.rows)});
    auto dy = tensor(output_grad,
        dextents<int32_t, 2>{int32_t(p.output), int32_t(p.rows)});
    auto dw = tensor(weight_grad,
        dextents<int32_t, 2>{int32_t(p.output), int32_t(p.input)});
    auto x_tile = x.slice(tile.y * 64, 0);
    auto dy_tile = dy.slice(tile.x * 64, 0);
    auto dw_tile = dw.slice(tile.x * 64, tile.y * 64);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(x_tile), decltype(dy_tile), float>();
    result.load(dw_tile);
    op.run(x_tile, dy_tile, result);
    result.store(dw_tile);
}

inline void train_add(device atomic_uint* address, float value) {
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &expected, as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed, memory_order_relaxed)) {}
}

kernel void dense_input_grad(
    device const half* weights [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device atomic_uint* input_grad [[buffer(2)]],
    constant DenseShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.input) return;
    const uint row = gid / p.input;
    const uint input = gid % p.input;
    float value = 0.0f;
    for (uint output = 0; output < p.output; ++output)
        value = fma(output_grad[ulong(row) * p.output + output],
                    float(weights[ulong(input) * p.output + output]), value);
    train_add(input_grad + gid, value);
}

kernel void dense_weight_grad(
    device const half* input [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device float* weight_grad [[buffer(2)]],
    constant DenseShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.input * p.output) return;
    const uint input_column = gid / p.output;
    const uint output_column = gid % p.output;
    float value = 0.0f;
    for (uint row = 0; row < p.rows; ++row)
        value = fma(float(input[ulong(row) * p.input + input_column]),
                    output_grad[ulong(row) * p.output + output_column], value);
    weight_grad[gid] += value;
}

struct RopeGradShape { uint rows, row_start, context, pairs; };

kernel void rope_grad(
    device float* qkv_grad [[buffer(0)]],
    device const float2* table [[buffer(1)]],
    constant RopeGradShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint qkv_width = 640;
    constexpr uint rope_heads = 9;
    const uint rotations = p.rows * rope_heads * p.pairs;
    if (gid >= rotations) return;
    const uint row = gid / (rope_heads * p.pairs);
    const uint item = gid % (rope_heads * p.pairs);
    const uint head = item / p.pairs;
    const uint pair = item % p.pairs;
    const uint position = (p.row_start + row) % p.context;
    const float2 phase = table[ulong(position) * p.pairs + pair];
    const ulong offset = ulong(row) * qkv_width + head * head_dim + pair;
    const float2 value = float2(qkv_grad[offset], qkv_grad[offset + p.pairs]);
    qkv_grad[offset] = fma(value.y, phase.y, value.x * phase.x);
    qkv_grad[offset + p.pairs] = fma(-value.x, phase.y, value.y * phase.x);
}

struct EmbedGradShape { uint rows, width, vocabulary, patch; };

kernel void embed_grad(
    device const half* readout [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device const half* positions [[buffer(2)]],
    device const float* output_grad [[buffer(3)]],
    device atomic_uint* readout_grad [[buffer(4)]],
    device atomic_uint* position_grad [[buffer(5)]],
    device atomic_uint* mask_grad [[buffer(6)]],
    constant EmbedGradShape& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.width) return;
    const uint row = gid / p.width;
    const uint dimension = gid % p.width;
    const uint token = tokens[row];
    if (token < p.vocabulary) {
        train_add(
            readout_grad + ulong(dimension) * p.vocabulary + token,
            output_grad[gid]);
    }
}

struct ReadoutGradShape {
    uint rows;
    uint width;
    uint vocab;
    uint start;
    float scale;
};

kernel void readout_grad(
    device const half* hidden [[buffer(0)]],
    device const half* readout [[buffer(1)]],
    device const uint* expected [[buffer(2)]],
    device const uchar* mask [[buffer(3)]],
    device float* hidden_grad [[buffer(4)]],
    device atomic_uint* readout_grad [[buffer(5)]],
    constant ReadoutGradShape& p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows || mask[row] == 0) return;
    const uint target = expected[row];
    if (target >= p.vocab) return;
    const ulong base = ulong(p.start + row) * p.width;
    threadgroup float shared_hidden[512];
    threadgroup float partial_max[8];
    threadgroup float partial_sum[8];
    threadgroup float row_max;
    threadgroup float row_sum;
    for (uint d = tid; d < p.width; d += 256) {
        shared_hidden[d] = float(hidden[base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = -INFINITY;
    for (uint col = tid; col < p.vocab; col += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < p.width; ++d) {
            dot = fma(shared_hidden[d], float(readout[ulong(d) * p.vocab + col]), dot);
        }
        local_max = max(local_max, float(half(dot)));
    }
    local_max = simd_max(local_max);
    if (lane == 0) partial_max[simdgroup] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float val = lane < 8 ? partial_max[lane] : -INFINITY;
        const float total = simd_max(val);
        if (lane == 0) row_max = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint col = tid; col < p.vocab; col += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < p.width; ++d) {
            dot = fma(shared_hidden[d], float(readout[ulong(d) * p.vocab + col]), dot);
        }
        local_sum += exp(float(half(dot)) - row_max);
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) partial_sum[simdgroup] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float val = lane < 8 ? partial_sum[lane] : 0.0f;
        const float total = simd_sum(val);
        if (lane == 0) row_sum = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / row_sum;
    for (uint col = tid; col < p.vocab; col += 256) {
        float dot = 0.0f;
        for (uint d = 0; d < p.width; ++d) {
            dot = fma(shared_hidden[d], float(readout[ulong(d) * p.vocab + col]), dot);
        }
        const float prob = exp(float(half(dot)) - row_max) * inv_sum;
        const float dl = (prob - float(col == target)) * p.scale;
        for (uint d = 0; d < p.width; ++d) {
            const ulong idx = ulong(d) * p.vocab + col;
            train_add(readout_grad + idx, shared_hidden[d] * dl);
        }
    }
    for (uint d = tid; d < p.width; d += 256) {
        float dx = 0.0f;
        for (uint col = 0; col < p.vocab; ++col) {
            float dot = 0.0f;
            for (uint k = 0; k < p.width; ++k) {
                dot = fma(shared_hidden[k], float(readout[ulong(k) * p.vocab + col]), dot);
            }
            const float prob = exp(float(half(dot)) - row_max) * inv_sum;
            const float dl = (prob - float(col == target)) * p.scale;
            dx = fma(dl, float(readout[ulong(d) * p.vocab + col]), dx);
        }
        hidden_grad[base + d] = dx;
    }
}

kernel void replicate_grad(
    device const float* stream_grad [[buffer(0)]],
    device float* input_grad [[buffer(1)]],
    constant DenseShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.input) return;
    const uint row = gid / p.input;
    const uint column = gid % p.input;
    const ulong base = ulong(row) * 4 * p.input + column;
    input_grad[gid] = stream_grad[base] + stream_grad[base + p.input]
        + stream_grad[base + 2 * p.input] + stream_grad[base + 3 * p.input];
}
