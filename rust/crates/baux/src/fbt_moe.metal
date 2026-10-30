#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct RopeShape {
    uint rows;
    uint row_start;
    uint context;
    uint pairs;
};

// Rotate every query head and the shared key head in place. Values remain
// unchanged. The table stores (cos, sin) for each sequence-local position and
// pair, avoiding transcendental work in the forward pass.
kernel void fbt_moe_rope(
    device half* qkv [[buffer(0)]],
    device const float2* table [[buffer(1)]],
    constant RopeShape& p [[buffer(2)]],
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
    const float2 value = float2(qkv[offset], qkv[offset + p.pairs]);
    qkv[offset] = half(fma(-value.y, phase.y, value.x * phase.x));
    qkv[offset + p.pairs] = half(fma(value.x, phase.y, value.y * phase.x));
}

struct MoeShape {
    uint rows;
    uint width;
    uint experts;
    uint rows_per_expert;
    uint expert_width;
};

kernel void fbt_moe_balanced_gate(
    device const half* input [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device half* gates [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = gid >> 5u;
    if (token >= p.rows) return;
    const uint expert = token % p.experts;
    float value = 0.0f;
    for (uint column = lane; column < p.width; column += 32u) {
        value += float(input[ulong(token) * p.width + column]) *
            float(router[ulong(column) * p.experts + expert]);
    }
    value = simd_sum(value);
    if (lane == 0u) {
        const float e = exp(-abs(value));
        gates[token] = half(value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
    }
}

kernel void fbt_moe_group_balanced(
    device const half* input [[buffer(0)]],
    device half* grouped [[buffer(1)]],
    constant MoeShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    const uint expert = token % p.experts;
    const uint row = token / p.experts;
    grouped[(ulong(expert) * p.rows_per_expert + row) * p.width + column] = input[gid];
}

kernel void fbt_moe_swiglu(
    device const half* gate_up [[buffer(0)]],
    device half* activation [[buffer(1)]],
    constant MoeShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.rows) * p.expert_width;
    if (gid >= total) return;
    const uint row = gid / p.expert_width;
    const uint column = gid % p.expert_width;
    const ulong base = ulong(row) * 2u * p.expert_width;
    const float gate = float(gate_up[base + column]);
    const float e = exp(-abs(gate));
    const float sigmoid = gate >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    activation[gid] = half(gate * sigmoid * float(gate_up[base + p.expert_width + column]));
}

kernel void fbt_moe_ungroup_residual(
    device const half* input [[buffer(0)]],
    device const half* grouped [[buffer(1)]],
    device const half* gates [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MoeShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint token = gid / p.width;
    const uint column = gid % p.width;
    const uint expert = token % p.experts;
    const uint row = token / p.experts;
    const ulong source = (ulong(expert) * p.rows_per_expert + row) * p.width + column;
    output[gid] = half(float(input[gid]) + float(gates[token]) * float(grouped[source]));
}

kernel void fbt_moe_feedback_fuse(
    device const half* state [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const float value = float(gate[gid]);
    const float e = exp(-abs(value));
    const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    output[gid] = half(float(state[gid]) * sigmoid);
}

// Mantissa-preserving log(1 + x) for x >= 0 (Goldberg 1991, Theorem 4;
// Barron et al., arXiv:2610.31642).
METAL_FUNC float stable_log1p(float x) {
    const float u = 1.0f + x;
    if (u <= 1.0f) return max(x, 0.0f);
    if (u - 1.0f == x) return log(u);
    return log(u) * (x / (u - 1.0f));
}

kernel void fbt_moe_cross_entropy(
    device const half* logits [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant uint& rows [[buffer(3)]], constant uint& vocabulary [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup uint peaks[8];
    threadgroup float shared;
    if (row >= rows) return;
    const ulong base = ulong(row) * vocabulary;
    float local_max = -INFINITY;
    for (uint column = tid; column < vocabulary; column += 256)
        local_max = max(local_max, float(logits[base + column]));
    const float group_max = simd_max(local_max);
    if (lane == 0) partial[simdgroup] = group_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : -INFINITY;
        const float maximum = simd_max(value);
        if (lane == 0) shared = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = shared;
    float local_tail = 0.0f;
    uint local_peaks = 0;
    for (uint column = tid; column < vocabulary; column += 256) {
        const float logit = float(logits[base + column]);
        if (logit == maximum) {
            local_peaks += 1;
        } else {
            local_tail += exp(logit - maximum);
        }
    }
    const float group_tail = simd_sum(local_tail);
    const uint group_peaks = simd_sum(local_peaks);
    if (lane == 0) {
        partial[simdgroup] = group_tail;
        peaks[simdgroup] = group_peaks;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : 0.0f;
        const uint peak_val = lane < 8 ? peaks[lane] : 0u;
        const float tail_total = simd_sum(value);
        const uint peak_total = simd_sum(peak_val);
        if (lane == 0) {
            const uint label = labels[row];
            const float total_tail = tail_total + float(max(peak_total, 1u) - 1u);
            losses[row] = (maximum - float(logits[base + label])) + stable_log1p(total_tail);
        }
    }
}

kernel void fbt_readout_loss_reduce(
    device const float4* partials [[buffer(0)]],
    device float* losses [[buffer(1)]],
    constant uint& tiles [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxima[4];
    threadgroup float tails[4];
    threadgroup uint peaks[4];
    threadgroup float targets[4];
    threadgroup float maximum;
    device const float4* row_partials = partials + ulong(row) * tiles;
    float thread_max = -INFINITY;
    float thread_tail = 0.0f;
    float thread_target = 0.0f;
    for (uint tile = tid; tile < tiles; tile += 128) {
        const float4 value = row_partials[tile];
        if (value.x > thread_max) {
            if (thread_max > -INFINITY) {
                const float scale = exp(thread_max - value.x);
                thread_tail = value.y + fma(thread_tail, scale, scale);
            } else {
                thread_tail = value.y;
            }
            thread_max = value.x;
        } else if (value.x > -INFINITY) {
            const float scale = exp(value.x - thread_max);
            thread_tail += fma(value.y, scale, scale);
        }
        thread_target += value.z;
    }
    const float local_max = simd_max(thread_max);
    if (lane == 0) maxima[simdgroup] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float global_max = simd_max(lane < 4 ? maxima[lane] : -INFINITY);
        if (lane == 0) maximum = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool is_peak = (thread_max == maximum) && (maximum > -INFINITY);
    float scaled_tail = 0.0f;
    if (is_peak) {
        scaled_tail = thread_tail;
    } else if (thread_max > -INFINITY) {
        const float scale = exp(thread_max - maximum);
        scaled_tail = fma(thread_tail, scale, scale);
    }
    const float tail_sum = simd_sum(scaled_tail);
    const uint peak_sum = simd_sum(is_peak ? 1u : 0u);
    const float target = simd_sum(thread_target);
    if (lane == 0) {
        tails[simdgroup] = tail_sum;
        peaks[simdgroup] = peak_sum;
        targets[simdgroup] = target;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float tail_total = simd_sum(lane < 4 ? tails[lane] : 0.0f);
        const uint peak_total = simd_sum(lane < 4 ? peaks[lane] : 0u);
        const float label = simd_sum(lane < 4 ? targets[lane] : 0.0f);
        if (lane == 0) {
            const float total_tail = tail_total + float(max(peak_total, 1u) - 1u);
            losses[row] = (maximum - label) + stable_log1p(total_tail);
        }
    }
}

kernel void fbt_readout_proposal_reduce(
    device const float4* partials [[buffer(0)]],
    device uint* tokens [[buffer(1)]],
    constant uint& tiles [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxima[4];
    threadgroup uint indices[4];
    device const float4* row_partials = partials + ulong(row) * tiles;
    float thread_max = -INFINITY;
    uint thread_index = UINT_MAX;
    for (uint tile = tid; tile < tiles; tile += 128) {
        const float4 packed = row_partials[tile];
        const uint candidate = as_type<uint>(packed.w);
        if (packed.x > thread_max || (packed.x == thread_max && candidate < thread_index)) {
            thread_max = packed.x;
            thread_index = candidate;
        }
    }
    const float local_max = simd_max(thread_max);
    const uint local_index = simd_min(
        thread_max == local_max ? thread_index : UINT_MAX);
    if (lane == 0) {
        maxima[simdgroup] = local_max;
        indices[simdgroup] = local_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? maxima[lane] : -INFINITY;
        const float maximum = simd_max(value);
        const uint index = simd_min(
            lane < 4 && value == maximum ? indices[lane] : UINT_MAX);
        if (lane == 0) tokens[row] = index;
    }
}

kernel void fbt_moe_rms(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float squared = 0.0f;
    for (uint column = tid; column < p.width; column += 128) {
        const float value = float(input[base + column]);
        squared = fma(value, value, squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < p.width; column += 128)
        output[base + column] = half(
            float(input[base + column]) * scale * float(weight[column]));
}

kernel void fbt_moe_embed(
    device const half* weights [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& vocabulary [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr uint rows = 8192;
    constexpr uint width = 512;
    if (gid >= rows * width) return;
    const uint row = gid / width;
    const uint column = gid % width;
    output[gid] = weights[ulong(column) * vocabulary + tokens[row]];
}

kernel void fbt_moe_sequence_loss(
    device const float* losses [[buffer(0)]],
    device const uchar* score_mask [[buffer(1)]],
    device float* scores [[buffer(2)]],
    uint sequence [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint context = 4096;
    threadgroup float partial[8];
    threadgroup uint counts[8];
    float sum = 0.0f;
    uint count = 0;
    const uint start = sequence * context;
    for (uint token = tid; token < context; token += 256) {
        const bool scored = score_mask[start + token] != 0;
        sum += scored ? losses[start + token] : 0.0f;
        count += uint(scored);
    }
    sum = simd_sum(sum);
    count = simd_sum(count);
    if (lane == 0) {
        partial[simdgroup] = sum;
        counts[simdgroup] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 8 ? partial[lane] : 0.0f;
        const uint count_value = lane < 8 ? counts[lane] : 0;
        const float total = simd_sum(value);
        const uint total_count = simd_sum(count_value);
        if (lane == 0) scores[sequence] = total / float(total_count);
    }
}

kernel void fbt_moe_residual(
    device const half* input [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MoeShape& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    output[gid] = half(float(input[gid]) + float(branch[gid]));
}

struct MhcShape {
    uint rows;
    uint width;
    uint architecture;
    uint reserved;
};

kernel void fbt_mhc_replicate(
    device const half* input [[buffer(0)]],
    device half* streams [[buffer(1)]],
    constant MhcShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint row = gid / p.width;
    const uint column = gid % p.width;
    const half value = input[gid];
    for (uint stream = 0; stream < 4; ++stream)
        streams[(ulong(row) * 4 + stream) * p.width + column] = value;
}

inline void fbt_mhc_finish(
    thread const float* raw,
    device float* result,
    device const half* control,
    uint architecture) {
    float input_total = 0.0f;
    for (uint index = 0; index < 4; ++index) {
        result[index] = 1.0f / (1.0f + exp(-raw[index]));
        input_total += result[index];
    }
    for (uint index = 0; index < 4; ++index)
        result[index] /= max(input_total, 1.0e-20f);

    if (architecture == 1) {
        for (uint output = 0; output < 4; ++output) {
            for (uint input = 0; input < 4; ++input) {
                const uint index = output * 4 + input;
                result[4 + index] = raw[4 + index] + (output == input ? 1.0f : 0.0f);
            }
        }
    } else {
        float transport[16];
        float maximum = raw[4];
        for (uint index = 1; index < 16; ++index)
            maximum = max(maximum, raw[4 + index]);
        for (uint index = 0; index < 16; ++index)
            transport[index] = exp(max(raw[4 + index] - maximum, -80.0f));
        for (uint iteration = 0; iteration < 20; ++iteration) {
            for (uint output = 0; output < 4; ++output) {
                float total = 0.0f;
                for (uint input = 0; input < 4; ++input)
                    total += transport[output * 4 + input];
                for (uint input = 0; input < 4; ++input)
                    transport[output * 4 + input] /= max(total, 1.0e-20f);
            }
            for (uint input = 0; input < 4; ++input) {
                float total = 0.0f;
                for (uint output = 0; output < 4; ++output)
                    total += transport[output * 4 + input];
                for (uint output = 0; output < 4; ++output)
                    transport[output * 4 + input] /= max(total, 1.0e-20f);
            }
        }
        const float lambda = clamp(float(control[3]), 0.0f, 1.0f);
        for (uint output = 0; output < 4; ++output) {
            for (uint input = 0; input < 4; ++input) {
                const uint index = output * 4 + input;
                const float identity = output == input ? 1.0f : 0.0f;
                result[4 + index] = mix(identity, transport[index], lambda);
            }
        }
    }
    for (uint index = 0; index < 4; ++index)
        result[20 + index] = 2.0f / (1.0f + exp(-raw[20 + index]));
}

kernel void fbt_mhc_predict(
    device const half* streams [[buffer(0)]],
    device const half* predictor [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device const half* control [[buffer(3)]],
    device float* coefficients [[buffer(4)]],
    constant MhcShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    constexpr uint kStreams = 4;
    constexpr uint kCoefficients = 24;
    constexpr uint kOutputsPerSimdgroup = 3;
    threadgroup float dot_sums[kCoefficients];
    threadgroup float raw[kCoefficients];
    threadgroup float norm_scale;
    const ulong state_base = ulong(row) * kStreams * p.width;
    const uint output_base = simdgroup * kOutputsPerSimdgroup;
    const bool any_control =
        control[0] != half(0.0f) ||
        control[1] != half(0.0f) ||
        control[2] != half(0.0f);
    float squared = 0.0f;
    float dots[kOutputsPerSimdgroup] = {0.0f, 0.0f, 0.0f};
    if (any_control) {
        for (uint input = lane; input < kStreams * p.width; input += 32) {
            const float value = float(streams[state_base + input]);
            if (simdgroup == 0) squared = fma(value, value, squared);
            for (uint item = 0; item < kOutputsPerSimdgroup; ++item) {
                const uint output = output_base + item;
                const uint group = output < 4 ? 0 : (output < 20 ? 1 : 2);
                if (control[group] != half(0.0f)) {
                    dots[item] = fma(
                        value,
                        float(predictor[ulong(input) * kCoefficients + output]),
                        dots[item]);
                }
            }
        }
    }
    for (uint item = 0; item < kOutputsPerSimdgroup; ++item) {
        const uint output = output_base + item;
        const float total = simd_sum(dots[item]);
        if (lane == 0) dot_sums[output] = total;
    }
    if (simdgroup == 0) {
        const float total = simd_sum(squared);
        if (lane == 0) {
            norm_scale = any_control
                ? rsqrt(total / float(kStreams * p.width) + 1.0e-5f)
                : 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < kCoefficients) {
        const uint group = tid < 4 ? 0 : (tid < 20 ? 1 : 2);
        raw[tid] = float(bias[tid]) + float(control[group]) * dot_sums[tid] * norm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float local_raw[kCoefficients];
    for (uint index = 0; index < kCoefficients; ++index)
        local_raw[index] = raw[index];
    device float* result = coefficients + ulong(row) * kCoefficients;
    fbt_mhc_finish(local_raw, result, control, p.architecture);
}

kernel void fbt_mhc_predict_rows(
    device const half* streams [[buffer(0)]],
    device const half* predictor [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device const half* control [[buffer(3)]],
    device float* coefficients [[buffer(4)]],
    constant MhcShape& p [[buffer(5)]],
    uint row_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint kRows = 64;
    constexpr uint kInput = 2048;
    constexpr uint kOutputs = 24;
    threadgroup half a_tile[kRows][36];
    threadgroup half b_tile[32][36];
    threadgroup float result_tile[kRows][36];
    threadgroup float norm_tile[kRows][8];
    const uint row_base = row_tile * kRows;
    const uint simdgroup = tid / 32;
    const uint simd_row = (simdgroup / 2) * 32;
    const uint simd_column = (simdgroup % 2) * 16;
    const bool active =
        control[0] != half(0.0f) ||
        control[1] != half(0.0f) ||
        control[2] != half(0.0f);
    simdgroup_float8x8 accumulators[4][2];
    float norm_accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j)
            accumulators[i][j] = simdgroup_float8x8(0.0f);

    if (active) {
        for (uint k_start = 0; k_start < kInput; k_start += 32) {
            for (uint step = 0; step < 4; ++step) {
                const uint index = tid + step * 128;
                const uint local_row = index / 8;
                const uint column4 = index % 8;
                const uint row = row_base + local_row;
                half4 value = half4(0.0h);
                if (row < p.rows) {
                    value = *reinterpret_cast<device const half4*>(
                        streams + ulong(row) * kInput + k_start + column4 * 4);
                    const float4 norm_value = float4(value);
                    norm_accumulators[step] += dot(norm_value, norm_value);
                }
                *reinterpret_cast<threadgroup half4*>(&a_tile[local_row][column4 * 4]) = value;
            }
            for (uint step = 0; step < 2; ++step) {
                const uint index = tid + step * 128;
                const uint local_row = index / 8;
                const uint column4 = index % 8;
                const uint column = column4 * 4;
                half4 value = half4(0.0h);
                if (column + 3 < kOutputs) {
                    value = *reinterpret_cast<device const half4*>(
                        predictor + ulong(k_start + local_row) * kOutputs + column);
                }
                *reinterpret_cast<threadgroup half4*>(&b_tile[local_row][column]) = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint depth = 0; depth < 32; depth += 8) {
                simdgroup_half8x8 a[4];
                simdgroup_half8x8 b[2];
                for (uint i = 0; i < 4; ++i)
                    simdgroup_load(a[i], &a_tile[simd_row + i * 8][depth], 36);
                for (uint j = 0; j < 2; ++j)
                    simdgroup_load(b[j], &b_tile[depth][simd_column + j * 8], 36);
                for (uint i = 0; i < 4; ++i)
                    for (uint j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(
                            accumulators[i][j], a[i], b[j], accumulators[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint step = 0; step < 4; ++step) {
            const uint index = tid + step * 128;
            norm_tile[index / 8][index % 8] = norm_accumulators[step];
        }
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 2; ++j)
                simdgroup_store(
                    accumulators[i][j],
                    &result_tile[simd_row + i * 8][simd_column + j * 8],
                    36);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < kRows) {
        const uint row = row_base + tid;
        if (row < p.rows) {
            float squared = 0.0f;
            if (active)
                for (uint column = 0; column < 8; ++column)
                    squared += norm_tile[tid][column];
            const float scale = active
                ? rsqrt(squared / float(kInput) + 1.0e-5f)
                : 1.0f;
            float raw[kOutputs];
            for (uint output = 0; output < kOutputs; ++output) {
                const uint group = output < 4 ? 0 : (output < 20 ? 1 : 2);
                const float dot = active ? result_tile[tid][output] : 0.0f;
                raw[output] = float(bias[output]) + float(control[group]) * dot * scale;
            }
            fbt_mhc_finish(
                raw,
                coefficients + ulong(row) * kOutputs,
                control,
                p.architecture);
        }
    }
}

kernel void fbt_mhc_mix_rms(
    device const half* streams [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* normalized [[buffer(3)]],
    constant MhcShape& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float norm_scale;
    if (row >= p.rows) return;
    const ulong stream_base = ulong(row) * 4 * p.width;
    const device float* a = coefficients + ulong(row) * 24;
    float values[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        float value = 0.0f;
        #pragma unroll
        for (uint stream = 0; stream < 4; ++stream)
            value = fma(a[stream], float(streams[stream_base + stream * p.width + column]), value);
        const half rounded = half(value);
        values[item] = float(rounded);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) norm_scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong output_base = ulong(row) * p.width;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        normalized[output_base + column] = half(values[item] * norm_scale * float(weight[column]));
    }
}

kernel void fbt_mhc_update(
    device const half* streams [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MhcShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * 4 * p.width;
    if (gid >= elements) return;
    const uint row = gid / (4 * p.width);
    const uint local = gid % (4 * p.width);
    const uint stream = local / p.width;
    const uint column = local % p.width;
    const ulong state_base = ulong(row) * 4 * p.width;
    const device float* coefficient = coefficients + ulong(row) * 24;
    float value = coefficient[20 + stream] * float(branch[ulong(row) * p.width + column]);
    for (uint source = 0; source < 4; ++source)
        value = fma(
            coefficient[4 + stream * 4 + source],
            float(streams[state_base + source * p.width + column]),
            value);
    output[gid] = half(value);
}

METAL_FUNC void mhc_transport(
    device const half* streams, float4 branch, device const float* a,
    device half* output, ulong base, uint width) {
    float4 source[4];
    for (uint s = 0; s < 4; ++s)
        source[s] = float4(*reinterpret_cast<device const half4*>(streams + base + s * width));
    for (uint destination = 0; destination < 4; ++destination) {
        float4 value = a[20 + destination] * branch;
        for (uint s = 0; s < 4; ++s)
            value = fma(a[4 + destination * 4 + s], source[s], value);
        *reinterpret_cast<device half4*>(output + base + destination * width) = half4(value);
    }
}

// A thread owns four columns across all destinations. Each source vector and
// branch is loaded once; the scalar kernel's ordered float FMAs are preserved.
kernel void fbt_mhc_update_rows(
    device const half* streams [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MhcShape& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * (p.width / 4)) return;
    const uint row = gid / (p.width / 4);
    const uint column = gid % (p.width / 4) * 4;
    const ulong base = ulong(row) * 4 * p.width + column;
    const device float* a = coefficients + ulong(row) * 24;
    const float4 b = float4(*reinterpret_cast<device const half4*>(
        branch + ulong(row) * p.width + column));
    mhc_transport(streams, b, a, output, base, p.width);
}

// Keep the branch for backward replay, but transport its rounded values
// immediately instead of reading them in a second device-wide dispatch.
kernel void combine_mhc(
    device const half* streams [[buffer(0)]],
    device half* branch [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const half* route_weights [[buffer(4)]],
    device const uint* packed_rows [[buffer(5)]],
    device const half* routed [[buffer(6)]],
    device const half* shared [[buffer(7)]],
    constant MhcShape& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * (p.width / 4)) return;
    const uint row = gid / (p.width / 4);
    const uint column = gid % (p.width / 4) * 4;
    const ulong destination = ulong(row) * p.width + column;
    float4 value = float4(*reinterpret_cast<device const half4*>(shared + destination));
    for (uint route = 0; route < 3; ++route) {
        const ulong assignment = ulong(row) * 3 + route;
        const ulong source = ulong(packed_rows[assignment]) * p.width + column;
        value = fma(float4(float(route_weights[assignment])),
            float4(*reinterpret_cast<device const half4*>(routed + source)), value);
    }
    volatile half4 rounded = half4(value);
    *reinterpret_cast<device half4*>(branch + destination) = rounded;
    mhc_transport(streams, float4(rounded), coefficients + ulong(row) * 24,
        output, ulong(row) * 4 * p.width + column, p.width);
}

kernel void fbt_mhc_mean_rms(
    device const half* streams [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* normalized [[buffer(2)]],
    constant MhcShape& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float norm_scale;
    if (row >= p.rows) return;
    const ulong stream_base = ulong(row) * 4 * p.width;
    float values[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        float value = 0.0f;
        #pragma unroll
        for (uint stream = 0; stream < 4; ++stream)
            value += 0.25f * float(streams[stream_base + stream * p.width + column]);
        const half rounded = half(value);
        values[item] = float(rounded);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) norm_scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong output_base = ulong(row) * p.width;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        normalized[output_base + column] = half(values[item] * norm_scale * float(weight[column]));
    }
}

kernel void fbt_moe_residual_rms(
    device const half* input [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* residual [[buffer(3)]],
    device half* normalized [[buffer(4)]],
    constant MoeShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float values[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const half value = half(float(input[base + column]) + float(branch[base + column]));
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        residual[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}

kernel void fbt_moe_ungroup_residual_rms(
    device const half* input [[buffer(0)]],
    device const half* grouped [[buffer(1)]],
    device const half* gates [[buffer(2)]],
    device const half* weight [[buffer(3)]],
    device half* output [[buffer(4)]],
    device half* normalized [[buffer(5)]],
    constant MoeShape& p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const uint expert = row % p.experts;
    const uint expert_row = row / p.experts;
    const ulong base = ulong(row) * p.width;
    const ulong source =
        (ulong(expert) * p.rows_per_expert + expert_row) * p.width;
    const float gate = float(gates[row]);
    float values[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const half value = half(float(input[base + column])
            + gate * float(grouped[source + column]));
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        output[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}

kernel void fbt_moe_feedback_fuse_rms(
    device const half* state [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    device half* normalized [[buffer(4)]],
    constant MoeShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    threadgroup float scale;
    if (row >= p.rows) return;
    const ulong base = ulong(row) * p.width;
    float values[4];
    float squared = 0.0f;
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        const float gate_value = float(gate[base + column]);
        const float e = exp(-abs(gate_value));
        const float sigmoid = gate_value >= 0.0f
            ? 1.0f / (1.0f + e)
            : e / (1.0f + e);
        const half value = half(float(state[base + column]) * sigmoid);
        values[item] = float(value);
        squared = fma(values[item], values[item], squared);
    }
    squared = simd_sum(squared);
    if (lane == 0) partial[simdgroup] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float value = lane < 4 ? partial[lane] : 0.0f;
        const float total = simd_sum(value);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint item = 0; item < 4; ++item) {
        const uint column = tid + item * 128;
        output[base + column] = half(values[item]);
        normalized[base + column] = half(values[item] * scale * float(weight[column]));
    }
}
