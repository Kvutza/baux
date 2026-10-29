#include <metal_stdlib>
using namespace metal;

struct MhcGradShape { uint rows, width, architecture, reserved; };

inline void mhc_add(device atomic_uint* address, float value) {
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &expected, as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed, memory_order_relaxed)) {}
}

kernel void mhc_mean_rms_grad(
    device const half* streams [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    device atomic_uint* stream_grad [[buffer(3)]],
    device atomic_uint* weight_grad [[buffer(4)]],
    constant MhcGradShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    threadgroup float partial[4];
    threadgroup float scale;
    threadgroup float correction;
    const ulong stream_base = ulong(row) * 4 * p.width;
    const ulong output_base = ulong(row) * p.width;
    float squares = 0.0f;
    float product = 0.0f;
    for (uint column = tid; column < p.width; column += 128) {
        float x = 0.0f;
        for (uint stream = 0; stream < 4; ++stream)
            x += 0.25f * float(streams[stream_base + stream * p.width + column]);
        x = float(half(x));
        squares = fma(x, x, squares);
        product = fma(output_grad[output_base + column] * float(weight[column]), x, product);
    }
    squares = simd_sum(squares);
    product = simd_sum(product);
    if (lane == 0) partial[simd] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[simd] = product;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) correction = total * scale * scale * scale / float(p.width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < p.width; column += 128) {
        float x = 0.0f;
        for (uint stream = 0; stream < 4; ++stream)
            x += 0.25f * float(streams[stream_base + stream * p.width + column]);
        x = float(half(x));
        const float dy = output_grad[output_base + column];
        const float dx = scale * dy * float(weight[column]) - x * correction;
        for (uint stream = 0; stream < 4; ++stream)
            mhc_add(stream_grad + stream_base + stream * p.width + column, 0.25f * dx);
        mhc_add(weight_grad + column, dy * x * scale);
    }
}

kernel void mhc_mix_rms_grad(
    device const half* streams [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const float* output_grad [[buffer(3)]],
    device atomic_uint* stream_grad [[buffer(4)]],
    device atomic_uint* coefficient_grad [[buffer(5)]],
    device atomic_uint* weight_grad [[buffer(6)]],
    constant MhcGradShape& p [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    threadgroup float partial[4];
    threadgroup float scale;
    threadgroup float correction;
    const ulong stream_base = ulong(row) * 4 * p.width;
    const ulong output_base = ulong(row) * p.width;
    const device float* a = coefficients + ulong(row) * 24;
    float squares = 0.0f;
    float product = 0.0f;
    for (uint column = tid; column < p.width; column += 128) {
        float x = 0.0f;
        for (uint stream = 0; stream < 4; ++stream)
            x = fma(a[stream], float(streams[stream_base + stream * p.width + column]), x);
        x = float(half(x));
        squares = fma(x, x, squares);
        product = fma(output_grad[output_base + column] * float(weight[column]), x, product);
    }
    squares = simd_sum(squares);
    product = simd_sum(product);
    if (lane == 0) partial[simd] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) scale = rsqrt(total / float(p.width) + 1.0e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[simd] = product;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane == 0) correction = total * scale * scale * scale / float(p.width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < p.width; column += 128) {
        float x = 0.0f;
        for (uint stream = 0; stream < 4; ++stream)
            x = fma(a[stream], float(streams[stream_base + stream * p.width + column]), x);
        x = float(half(x));
        const float dy = output_grad[output_base + column];
        const float dx = scale * dy * float(weight[column]) - x * correction;
        for (uint stream = 0; stream < 4; ++stream) {
            const float state = float(streams[stream_base + stream * p.width + column]);
            mhc_add(stream_grad + stream_base + stream * p.width + column, a[stream] * dx);
            mhc_add(coefficient_grad + ulong(row) * 24 + stream, state * dx);
        }
        mhc_add(weight_grad + column, dy * x * scale);
    }
}

kernel void mhc_update_grad(
    device const half* streams [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    device const float* coefficients [[buffer(2)]],
    device const float* output_grad [[buffer(3)]],
    device atomic_uint* stream_grad [[buffer(4)]],
    device atomic_uint* branch_grad [[buffer(5)]],
    device atomic_uint* coefficient_grad [[buffer(6)]],
    constant MhcGradShape& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.rows) * p.width;
    if (gid >= elements) return;
    const uint row = gid / p.width;
    const uint column = gid % p.width;
    const ulong base = ulong(row) * 4 * p.width;
    const device float* a = coefficients + ulong(row) * 24;
    float db = 0.0f;
    for (uint destination = 0; destination < 4; ++destination) {
        const float dy = output_grad[base + destination * p.width + column];
        db = fma(a[20 + destination], dy, db);
        mhc_add(coefficient_grad + ulong(row) * 24 + 20 + destination,
                dy * float(branch[ulong(row) * p.width + column]));
        for (uint source = 0; source < 4; ++source) {
            mhc_add(stream_grad + base + source * p.width + column,
                    a[4 + destination * 4 + source] * dy);
            mhc_add(coefficient_grad + ulong(row) * 24 + 4 + destination * 4 + source,
                    dy * float(streams[base + source * p.width + column]));
        }
    }
    mhc_add(branch_grad + ulong(row) * p.width + column, db);
}

inline void sinkhorn_start(thread const float* raw, thread float* state) {
    float maximum = raw[4];
    for (uint i = 1; i < 16; ++i) maximum = max(maximum, raw[4 + i]);
    for (uint i = 0; i < 16; ++i) state[i] = exp(max(raw[4 + i] - maximum, -80.0f));
}

inline void sinkhorn_row(thread float* state) {
    for (uint row = 0; row < 4; ++row) {
        float total = 0.0f;
        for (uint column = 0; column < 4; ++column) total += state[row * 4 + column];
        for (uint column = 0; column < 4; ++column) state[row * 4 + column] /= max(total, 1.0e-20f);
    }
}

inline void sinkhorn_column(thread float* state) {
    for (uint column = 0; column < 4; ++column) {
        float total = 0.0f;
        for (uint row = 0; row < 4; ++row) total += state[row * 4 + column];
        for (uint row = 0; row < 4; ++row) state[row * 4 + column] /= max(total, 1.0e-20f);
    }
}

inline void sinkhorn_state(thread const float* raw, uint iterations, thread float* state) {
    sinkhorn_start(raw, state);
    for (uint i = 0; i < iterations; ++i) {
        sinkhorn_row(state);
        sinkhorn_column(state);
    }
}

kernel void mhc_predict_grad(
    device const half* streams [[buffer(0)]],
    device const half* predictor [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device const half* control [[buffer(3)]],
    device const float* coefficient_grad [[buffer(4)]],
    device atomic_uint* stream_grad [[buffer(5)]],
    device atomic_uint* predictor_grad [[buffer(6)]],
    device atomic_uint* bias_grad [[buffer(7)]],
    device atomic_uint* control_grad [[buffer(8)]],
    constant MhcGradShape& p [[buffer(9)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    constexpr uint input_size = 2048;
    constexpr uint outputs = 24;
    const ulong state_base = ulong(row) * input_size;
    float squared = 0.0f;
    for (uint input = 0; input < input_size; ++input) {
        const float value = float(streams[state_base + input]);
        squared = fma(value, value, squared);
    }
    const bool active = control[0] != half(0.0f)
        || control[1] != half(0.0f) || control[2] != half(0.0f);
    const float scale = active ? rsqrt(squared / float(input_size) + 1.0e-5f) : 1.0f;
    float dots[outputs];
    float raw[outputs];
    for (uint output = 0; output < outputs; ++output) {
        float dot = 0.0f;
        if (active)
            for (uint input = 0; input < input_size; ++input)
                dot = fma(float(streams[state_base + input]),
                          float(predictor[ulong(input) * outputs + output]), dot);
        dots[output] = dot;
        const uint group = output < 4 ? 0 : (output < 20 ? 1 : 2);
        raw[output] = float(bias[output]) + float(control[group]) * dot * scale;
    }

    const device float* dc = coefficient_grad + ulong(row) * outputs;
    float draw[outputs];
    for (uint i = 0; i < outputs; ++i) draw[i] = 0.0f;
    float sigmoid[4];
    float total = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        sigmoid[i] = 1.0f / (1.0f + exp(-raw[i]));
        total += sigmoid[i];
    }
    float center = 0.0f;
    for (uint i = 0; i < 4; ++i) center += dc[i] * sigmoid[i] / total;
    for (uint i = 0; i < 4; ++i)
        draw[i] = (dc[i] - center) * sigmoid[i] * (1.0f - sigmoid[i]) / total;

    if (p.architecture == 1) {
        for (uint i = 0; i < 16; ++i) draw[4 + i] = dc[4 + i];
    } else {
        float state[16];
        sinkhorn_state(raw, 20, state);
        const float lambda = clamp(float(control[3]), 0.0f, 1.0f);
        float gradient[16];
        float dlambda = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            const float identity = i / 4 == i % 4 ? 1.0f : 0.0f;
            gradient[i] = lambda * dc[4 + i];
            dlambda += dc[4 + i] * (state[i] - identity);
        }
        if (float(control[3]) > 0.0f && float(control[3]) < 1.0f)
            mhc_add(control_grad + 3, dlambda);
        for (int iteration = 19; iteration >= 0; --iteration) {
            float before[16];
            float after_row[16];
            sinkhorn_state(raw, uint(iteration), before);
            for (uint i = 0; i < 16; ++i) after_row[i] = before[i];
            sinkhorn_row(after_row);
            float next[16];
            for (uint column = 0; column < 4; ++column) {
                float sum = 0.0f;
                float dot = 0.0f;
                for (uint r = 0; r < 4; ++r) {
                    sum += after_row[r * 4 + column];
                    dot += gradient[r * 4 + column]
                        * (after_row[r * 4 + column] / sum);
                }
                // Recompute with the complete denominator.
                dot = 0.0f;
                for (uint r = 0; r < 4; ++r)
                    dot += gradient[r * 4 + column] * after_row[r * 4 + column] / sum;
                for (uint r = 0; r < 4; ++r)
                    next[r * 4 + column] = (gradient[r * 4 + column] - dot) / sum;
            }
            for (uint r = 0; r < 4; ++r) {
                float sum = 0.0f;
                float dot = 0.0f;
                for (uint column = 0; column < 4; ++column) sum += before[r * 4 + column];
                for (uint column = 0; column < 4; ++column)
                    dot += next[r * 4 + column] * before[r * 4 + column] / sum;
                for (uint column = 0; column < 4; ++column)
                    gradient[r * 4 + column] = (next[r * 4 + column] - dot) / sum;
            }
        }
        float maximum = raw[4];
        uint maximum_index = 0;
        for (uint i = 1; i < 16; ++i) {
            if (raw[4 + i] > maximum) { maximum = raw[4 + i]; maximum_index = i; }
        }
        float dmaximum = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            if (raw[4 + i] - maximum > -80.0f) {
                const float value = gradient[i] * exp(raw[4 + i] - maximum);
                draw[4 + i] += value;
                dmaximum -= value;
            }
        }
        draw[4 + maximum_index] += dmaximum;
    }
    for (uint i = 0; i < 4; ++i) {
        const float s = 1.0f / (1.0f + exp(-raw[20 + i]));
        draw[20 + i] = dc[20 + i] * 2.0f * s * (1.0f - s);
    }

    float norm_term = 0.0f;
    for (uint output = 0; output < outputs; ++output) {
        const uint group = output < 4 ? 0 : (output < 20 ? 1 : 2);
        mhc_add(bias_grad + output, draw[output]);
        mhc_add(control_grad + group, draw[output] * dots[output] * scale);
        norm_term += draw[output] * float(control[group]) * dots[output];
        const float factor = draw[output] * float(control[group]) * scale;
        for (uint input = 0; input < input_size; ++input)
            mhc_add(predictor_grad + ulong(input) * outputs + output,
                    factor * float(streams[state_base + input]));
    }
    const float norm_correction = norm_term * scale * scale * scale / float(input_size);
    for (uint input = 0; input < input_size; ++input) {
        float ds = -float(streams[state_base + input]) * norm_correction;
        for (uint output = 0; output < outputs; ++output) {
            const uint group = output < 4 ? 0 : (output < 20 ? 1 : 2);
            ds = fma(draw[output] * float(control[group]) * scale,
                     float(predictor[ulong(input) * outputs + output]), ds);
        }
        mhc_add(stream_grad + state_base + input, ds);
    }
}
