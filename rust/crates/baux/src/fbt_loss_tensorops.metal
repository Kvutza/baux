#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct ReadoutShape {
    uint rows;
    uint vocab;
};
struct ProposalShape {
    uint rows;
    uint vocab;
    uint row_start;
    uint context;
};

// Track non-maximal log-sum-exp tail mass without adding +1.0f for the peak,
// avoiding Softmax Collapse (arXiv:2610.31642) prior to log1p reduction.
METAL_FUNC void online_tail_push(float logit, thread float& maximum, thread float& tail) {
    if (logit > maximum) {
        if (maximum > -INFINITY) {
            const float scale = exp(maximum - logit);
            tail = fma(tail, scale, scale);
        }
        maximum = logit;
    } else if (logit > -INFINITY) {
        tail += exp(logit - maximum);
    }
}

// Preserve FP16 logit rounding and the complete vocabulary. Only loss
// sufficient statistics cross the device-memory boundary.
kernel void fbt_readout_loss_tiles(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device const uint* labels [[buffer(2)]],
    device float4* partials [[buffer(3)]],
    constant ReadoutShape& p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint stride = 68;
    threadgroup half logits[128 * stride];
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = group.y * 128;
    const uint first_column = group.x * 64;
    const uint rows = min(128u, p.rows - first_row);
    auto a = tensor(input + ulong(first_row) * 512,
                    dextents<int32_t, 2>{512, int32_t(rows)});
    auto b = tensor(weights + first_column,
                    dextents<int32_t, 2>{64, 512},
                    array<int32_t, 2>{1, int32_t(p.vocab)});
    auto c = tensor(logits,
                    dextents<int32_t, 2>{64, int32_t(rows)},
                    array<int32_t, 2>{1, int32_t(stride)});
    auto values = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    op.run(a, b, values);
    values.store(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= rows) return;
    const uint row = first_row + tid;
    const threadgroup half* row_logits = logits + tid * stride;
    float maximum = -INFINITY;
    float tail = 0.0f;
    uint best_index = first_column;
    for (uint column = 0; column < 64; ++column) {
        const float logit = float(row_logits[column]);
        if (logit > maximum) {
            best_index = first_column + column;
        }
        online_tail_push(logit, maximum, tail);
    }
    const uint label = labels[row];
    const float target = label >= first_column && label < first_column + 64
        ? float(row_logits[label - first_column]) : 0.0f;
    partials[ulong(row) * (p.vocab / 64) + group.x] =
        float4(maximum, tail, target, as_type<float>(best_index));
}

METAL_FUNC ulong fbt_decode_mix(ulong value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

// One full-vocabulary proposal per row without materializing [rows, vocab].
// The same absolute-position Gumbel draw makes repeated Jacobi verification
// deterministic and therefore capable of reproducing sampled AR decoding.
kernel void fbt_readout_proposal_tiles(
    device half* input [[buffer(0)]],
    device half* weights [[buffer(1)]],
    device float4* partials [[buffer(2)]],
    constant ProposalShape& p [[buffer(3)]],
    device const ulong* seeds [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    device const uint* labels [[buffer(6)]],
    device float4* loss_partials [[buffer(7)]],
    constant bool& score_targets [[buffer(8)]],
#ifdef DENOISE
    device float2* moments [[buffer(9)]],
#endif
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint stride = 68;
    threadgroup half logits[128 * stride];
    constexpr auto descriptor = matmul2d_descriptor(128, 64, 512);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    const uint first_row = group.y * 128;
    const uint first_column = group.x * 64;
    const uint rows = min(128u, p.rows - first_row);
    auto a = tensor(input + ulong(first_row) * 512,
                    dextents<int32_t, 2>{512, int32_t(rows)});
    auto b = tensor(weights + first_column,
                    dextents<int32_t, 2>{64, 512},
                    array<int32_t, 2>{1, int32_t(p.vocab)});
    auto c = tensor(logits,
                    dextents<int32_t, 2>{64, int32_t(rows)},
                    array<int32_t, 2>{1, int32_t(stride)});
    auto values = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), half>();
    op.run(a, b, values);
    values.store(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= rows) return;
    const uint row = first_row + tid;
    const threadgroup half* row_logits = logits + tid * stride;
    const uint limit = min(64u, p.vocab - first_column);
    float best = -INFINITY;
    uint index = UINT_MAX;
    float maximum = -INFINITY;
    float tail = 0.0f;
#ifdef DENOISE
    float chosen_logit = 0.0f;
    float scaled_max = -INFINITY;
    for (uint column = 0; column < limit; ++column) {
        const uint token = first_column + column;
        const float logit = float(row_logits[column]);
        online_tail_push(logit, maximum, tail);
        scaled_max = max(scaled_max, logit / max(temperature, 1e-6f));
        float value = logit;
        if (temperature > 0.0f) {
            const uint absolute_row = max(1u, p.row_start + row) - 1;
            const uint sequence = absolute_row / p.context;
            const uint position = absolute_row % p.context;
            const ulong bits = fbt_decode_mix(
                seeds[sequence] ^ (ulong(position) << 32) ^ token);
            const float uniform = (float(uint(bits >> 41)) + 0.5f) / 8388608.0f;
            value -= temperature * log(-log(uniform));
        }
        if (value > best || (value == best && token < index)) {
            best = value;
            index = token;
            chosen_logit = logit;
        }
    }
    partials[ulong(row) * (p.vocab / 64) + group.x] =
        float4(best, as_type<float>(index), chosen_logit, as_type<float>(index));
    float scaled_sum = 0.0f;
    for (uint column = 0; column < 64; ++column)
        scaled_sum += exp(float(row_logits[column]) / max(temperature, 1e-6f) - scaled_max);
    moments[ulong(row) * (p.vocab / 64) + group.x] = float2(scaled_max, scaled_sum);
    if (score_targets) {
        const uint label = labels[p.row_start + row];
        const float target = label >= first_column && label < first_column + 64
            ? float(row_logits[label - first_column]) : 0.0f;
        loss_partials[ulong(row) * (p.vocab / 64) + group.x] =
            float4(maximum, tail, target, as_type<float>(index));
    }
#else
    if (temperature == 0.0f) {
        for (uint column = 0; column < limit; ++column) {
            const float logit = float(row_logits[column]);
            if (logit > maximum) {
                index = first_column + column;
            }
            if (score_targets) {
                online_tail_push(logit, maximum, tail);
            } else if (logit > maximum) {
                maximum = logit;
            }
        }
        if (score_targets) {
            const uint label = labels[p.row_start + row];
            const float target = label >= first_column && label < first_column + 64
                ? float(row_logits[label - first_column]) : 0.0f;
            loss_partials[ulong(row) * (p.vocab / 64) + group.x] =
                float4(maximum, tail, target, as_type<float>(index));
        } else {
            partials[ulong(row) * (p.vocab / 64) + group.x] =
                float4(maximum, as_type<float>(index), 0.0f, as_type<float>(index));
        }
    } else {
        const uint absolute_row = p.row_start + row;
        const uint sequence = absolute_row / p.context;
        const uint position = absolute_row % p.context;
        const ulong row_seed = seeds[sequence] ^ (ulong(position) << 32);
        for (uint column = 0; column < limit; ++column) {
            const uint token = first_column + column;
            const float logit = float(row_logits[column]);
            if (score_targets) {
                online_tail_push(logit, maximum, tail);
            }
            const ulong bits = fbt_decode_mix(row_seed ^ token);
            const float uniform = (float(uint(bits >> 41)) + 0.5f) / 8388608.0f;
            const float value = logit - temperature * log(-log(uniform));
            if (value > best || (value == best && token < index)) {
                best = value;
                index = token;
            }
        }
        partials[ulong(row) * (p.vocab / 64) + group.x] =
            float4(best, as_type<float>(index), 0.0f, as_type<float>(index));
        if (score_targets) {
            const uint label = labels[p.row_start + row];
            const float target = label >= first_column && label < first_column + 64
                ? float(row_logits[label - first_column]) : 0.0f;
            loss_partials[ulong(row) * (p.vocab / 64) + group.x] =
                float4(maximum, tail, target, as_type<float>(index));
        }
    }
#endif
}
