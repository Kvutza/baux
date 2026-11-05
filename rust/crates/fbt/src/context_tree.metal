#include <metal_stdlib>
using namespace metal;

// Projected, rotated chunk rows are packed Q(512), K(64), V(64).
// Only KV persists across chunks; queries remain in bounded scratch.
kernel void context_pack(
    device const half4* qkv [[buffer(0)]],
    device half4* kv [[buffer(1)]],
    device half4* queries [[buffer(2)]],
    constant uint2& range [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= range.y * 160) return;
    const uint row = gid / 160, vec = gid % 160;
    if (vec < 128) queries[ulong(row) * 128 + vec] = qkv[gid];
    else kv[ulong(range.x + row) * 32 + vec - 128] = qkv[gid];
}

// CSA2 Reuse mode: unpack only Q(512) when KV is shared from an anchor layer.
kernel void context_queries(
    device const half4* qkv [[buffer(0)]],
    device half4* queries [[buffer(1)]],
    constant uint2& range [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= range.y * 128) return;
    const uint row = gid / 128, vec = gid % 128;
    queries[ulong(row) * 128 + vec] = qkv[ulong(row) * 160 + vec];
}

// KV rows contain one 64-dimensional key followed by one value.
kernel void context_leaves(
    device const half* kv [[buffer(0)]],
    device half* tree [[buffer(1)]],
    constant uint2& range [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint dim [[thread_index_in_threadgroup]]) {
    const uint leaf = range.x + group;
    if (group >= range.y || dim >= 64) return;
    float sum = 0.0f;
    const ulong base = (ulong(leaf) << 13) + dim;
    _Pragma("clang loop unroll_count(8)")
    for (uint token = 0; token < 64; ++token)
        sum += float(kv[base + (ulong(token) << 7)]);
    tree[(ulong(leaf) << 6) + dim] = half(sum * 0.015625f);
}

// Single-threadgroup bottom-up closure over all dirty parent levels.
kernel void context_parents(
    device half* tree [[buffer(0)]],
    constant uint4& range [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint dim = tid & 63;
    const uint worker = tid >> 6;
    const uint workers = max(1u, threads >> 6);
    if (worker >= workers) return;
    uint count = range.x;
    uint first = range.y;
    uint end = range.z;
    uint child = 0;
    uint parent = count;
    while (count > 1) {
        first >>= 1;
        end = (end + 1) >> 1;
        for (uint node = first + worker; node < end; node += workers) {
            const ulong left = (ulong(child + 2 * node) << 6) + dim;
            const ulong output = (ulong(parent + node) << 6) + dim;
            tree[output] = half(0.5f * (float(tree[left]) + float(tree[left + 64])));
        }
        threadgroup_barrier(mem_flags::mem_device);
        child = parent;
        count >>= 1;
        parent += count;
    }
}
