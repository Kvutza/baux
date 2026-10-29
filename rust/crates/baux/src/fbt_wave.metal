#include <metal_stdlib>
using namespace metal;

// 8^6 is the exact vocabulary. Every non-leaf node is the normalized sum of
// the tied token readouts below it.
kernel void mhc4_centroids(
    device const half* readout [[buffer(0)]],
    device half* centroids [[buffer(1)]],
    constant WaveShape& p [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint dimensions = 64;
    const uint groups = 584;
    if (group_id >= dimensions * groups) return;
    threadgroup float partials[2];
    const uint dimension = group_id / groups;
    const uint group = group_id % groups;
    device const half* row_ptr = readout + ulong(dimension) * p.vocabulary;
    if (group < 512) {
        const uint block = group;
        device const half4* src4 =
            reinterpret_cast<device const half4*>(row_ptr + block * 512 + tid * 8);
        const float4 a = float4(src4[0]);
        const float4 b = float4(src4[1]);
        const float sum8 = (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w);
        const uint node4 = 4680 + block * 64 + tid;
        centroids[ulong(node4) * dimensions + dimension] = half(sum8 * rsqrt(8.0f));
        float sum64 = sum8;
        sum64 += simd_shuffle_down(sum64, 4);
        sum64 += simd_shuffle_down(sum64, 2);
        sum64 += simd_shuffle_down(sum64, 1);
        if ((tid & 7u) == 0) {
            const uint node3 = 584 + block * 8 + (tid >> 3);
            centroids[ulong(node3) * dimensions + dimension] = half(sum64 * rsqrt(64.0f));
        }
        const float sg_sum = simd_sum(sum8);
        if (lane == 0) partials[sg] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint node2 = 72 + block;
            centroids[ulong(node2) * dimensions + dimension] =
                half((partials[0] + partials[1]) * rsqrt(512.0f));
        }
    } else {
        const uint node = group - 512;
        const uint leaves = node < 8 ? 32768 : 4096;
        const uint first = node < 8 ? node * 32768 : (node - 8) * 4096;
        device const half4* src4 = reinterpret_cast<device const half4*>(row_ptr + first);
        const uint vecs = leaves >> 2;
        float sum = 0.0f;
        for (uint v = tid; v < vecs; v += 64) {
            const float4 x = float4(src4[v]);
            sum += (x.x + x.y) + (x.z + x.w);
        }
        const float sg_sum = simd_sum(sum);
        if (lane == 0) partials[sg] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            centroids[ulong(node) * dimensions + dimension] =
                half((partials[0] + partials[1]) * rsqrt(float(leaves)));
        }
    }
}

METAL_FUNC uint mhc4_offset(uint level) {
    constexpr uint offsets[5] = {0, 8, 72, 584, 4680};
    return offsets[level];
}

kernel void mhc4_readout(
    device const half* hidden [[buffer(0)]],
    device const half* qkv [[buffer(1)]],
    device const half* readout [[buffer(2)]],
    device const half* centroids [[buffer(3)]],
    device const uint* expected [[buffer(4)]],
    device uint* output [[buffer(5)]],
    device float* losses [[buffer(6)]],
    constant WaveShape& p [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint branch [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.tokens) return;
    threadgroup half query[64];
    threadgroup float logits[2][8];
    threadgroup uint selected;
    threadgroup float token_loss;
    const uint tid = branch * 32 + lane;
    if (tid < 64) {
        device const half4* hidden4 =
            reinterpret_cast<device const half4*>(hidden + ulong(row) * 512);
        float value = 0.0f;
        for (uint vec = 0; vec < 128; ++vec) {
            const float4 h = float4(hidden4[vec]);
            const ulong base = ulong(vec * 4) * 640 + tid;
            value = fma(h.x, float(qkv[base]), value);
            value = fma(h.y, float(qkv[base + 640]), value);
            value = fma(h.z, float(qkv[base + 1280]), value);
            value = fma(h.w, float(qkv[base + 1920]), value);
        }
        query[tid] = half(value * 2.82842712f);
    }
    if (branch == 0 && lane == 0) selected = 0;
    if (branch == 1 && lane == 0) token_loss = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (branch != 0 && p.score == 0) return;
    const uint target_token = p.score != 0 ? expected[row] : 0;
    const uint child = lane >> 2;
    const uint child_lane = lane & 3u;
    threadgroup const half4* query4 = reinterpret_cast<threadgroup const half4*>(query);
    for (uint level = 0; level < 6; ++level) {
        const uint parent = branch != 0
            ? target_token >> ((6 - level) * 3)
            : selected;
        const uint index = parent * 8 + child;
        float value = 0.0f;
        if (level < 5) {
            const uint node = mhc4_offset(level) + index;
            device const half4* centroid4 = reinterpret_cast<device const half4*>(
                centroids + ulong(node) * 64);
            for (uint vector = child_lane; vector < 16; vector += 4)
                value += dot(float4(query4[vector]), float4(centroid4[vector]));
        } else {
            for (uint d = child_lane; d < 64; d += 4)
                value = fma(float(query[d]),
                            float(readout[ulong(d) * p.vocabulary + index]), value);
        }
        value += simd_shuffle_down(value, 2);
        value += simd_shuffle_down(value, 1);
        if (child_lane == 0) logits[branch][child] = value;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (branch == 0 && lane == 0) {
            uint choice = 0;
            for (uint c = 1; c < 8; ++c)
                if (logits[0][c] > logits[0][choice]) choice = c;
            selected = selected * 8 + choice;
        } else if (branch == 1 && lane == 0) {
            uint max_child = 0;
            float maximum = logits[1][0];
            for (uint c = 1; c < 8; ++c) {
                if (logits[1][c] > maximum) {
                    maximum = logits[1][c];
                    max_child = c;
                }
            }
            float tail = 0.0f;
            for (uint c = 0; c < 8; ++c) {
                if (c != max_child) tail += exp(logits[1][c] - maximum);
            }
            const uint target_child = (target_token >> ((5 - level) * 3)) & 7u;
            const float u = 1.0f + tail;
            const float log1p_tail = u <= 1.0f
                ? max(tail, 0.0f)
                : (u - 1.0f == tail ? log(u) : log(u) * (tail / (u - 1.0f)));
            const float level_nll = (maximum - logits[1][target_child]) + log1p_tail;
            token_loss += p.score == 2u
                ? log(2.0f - exp(-max(level_nll, 0.0f)))
                : level_nll;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (branch == 0 && lane == 0) output[row] = selected;
    if (branch == 1 && lane == 0 && p.score != 0) losses[row] = token_loss;
}
