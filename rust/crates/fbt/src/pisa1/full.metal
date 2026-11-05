// Fixed-shape two-stage full-PISA prefill. pisa1.metal is concatenated before
// this source and provides the hierarchy constants and scoring primitives.

constant uint kPisaCandidates = 16;

kernel void fbt_pisa_full_clear(
    device atomic_uint* counts [[buffer(0)]],
    device atomic_uint* tile_count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint clusters = 2 * kLeaves;
    if (index < clusters)
        atomic_store_explicit(&counts[index], 0, memory_order_relaxed);
    if (index == 0) {
        atomic_store_explicit(tile_count, 0, memory_order_relaxed);
        atomic_store_explicit(tile_count + 1, 1, memory_order_relaxed);
        atomic_store_explicit(tile_count + 2, 1, memory_order_relaxed);
    }
}

kernel void fbt_pisa_full_intermediate(
    device const half* qkv [[buffer(0)]],
    device const half* pyramid [[buffer(1)]],
    device uint* leaf_candidates [[buffer(2)]],
    device atomic_uint* counts [[buffer(3)]],
    device float* leaf_scores [[buffer(4)]],
    constant uint2& row_range [[buffer(5)]],
    uint query [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = row_range.x + query;
    if (query >= row_range.y || row >= kRows) return;
    threadgroup float scores[kPisaCandidates];
    threadgroup uint candidates[kPisaCandidates];
    threadgroup uint chosen[kSelected];
    threadgroup uint expanded[kPisaCandidates];

    if (lane < kPisaCandidates) candidates[lane] = lane;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint position = row % kContext;
    const uint current = position / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const uint sequence = row / kContext;
    const ulong query_base = ulong(row) * kQkvWidth;
    const ulong tree_base = ulong(sequence) * kNodes * kHeadDim;
    for (int level = int(31 - clz(kLeaves)) - 4; level >= 1; --level) {
        if (lane < kPisaCandidates) {
            const uint node = candidates[lane];
            float score = -INFINITY;
            if (node != UINT_MAX) {
                const uint span = 1u << uint(level);
                const uint last = (node + 1) * span - 1;
                const bool forced = contains_leaf(node, uint(level), 0)
                    || contains_leaf(node, uint(level), previous)
                    || contains_leaf(node, uint(level), current);
                if (forced) {
                    score = INFINITY;
                } else if (last < current) {
                    score = pisa_intermediate(
                        qkv,
                        query_base,
                        pyramid,
                        tree_base,
                        node,
                        uint(level));
                }
            }
            scores[lane] = score;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (uint slot = 0; slot < kSelected; ++slot) {
                uint best = UINT_MAX;
                float best_score = -INFINITY;
                for (uint candidate = 0; candidate < kPisaCandidates; ++candidate) {
                    const float score = scores[candidate];
                    if (score > best_score
                        || (score == best_score && score > -INFINITY
                            && candidates[candidate] < candidates[best])) {
                        best = candidate;
                        best_score = score;
                    }
                }
                chosen[slot] = best == UINT_MAX ? UINT_MAX : candidates[best];
                if (best != UINT_MAX) scores[best] = -INFINITY;
            }
            for (uint slot = 0; slot < kSelected; ++slot) {
                const uint node = chosen[slot];
                expanded[2 * slot] = node == UINT_MAX ? UINT_MAX : 2 * node;
                expanded[2 * slot + 1] = node == UINT_MAX ? UINT_MAX : 2 * node + 1;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < kPisaCandidates) candidates[lane] = expanded[lane];
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < kPisaCandidates) {
        const uint leaf = candidates[lane];
        const ulong output = ulong(row) * kPisaCandidates + lane;
        leaf_candidates[output] = leaf;
        if (leaf == UINT_MAX || leaf > current) {
            leaf_scores[output] = -INFINITY;
        } else if (leaf == 0 || leaf == previous || leaf == current) {
            leaf_scores[output] = INFINITY;
        } else {
            const uint cluster = sequence * kLeaves + leaf;
            atomic_fetch_add_explicit(&counts[cluster], 1, memory_order_relaxed);
        }
    }
}

kernel void fbt_pisa_full_tiles(
    device const atomic_uint* counts [[buffer(0)]],
    device atomic_uint* tile_count [[buffer(1)]],
    device uint* offsets [[buffer(2)]],
    device atomic_uint* cursors [[buffer(3)]],
    device uint4* tiles [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    const uint clusters = 2 * kLeaves;
    uint reference_offset = 0;
    uint tile_offset = 0;
    for (uint cluster = 0; cluster < clusters; ++cluster) {
        const uint count = atomic_load_explicit(&counts[cluster], memory_order_relaxed);
        offsets[cluster] = reference_offset;
        atomic_store_explicit(&cursors[cluster], reference_offset, memory_order_relaxed);
        for (uint first = 0; first < count; first += PISA_QUERY_TILE) {
            tiles[tile_offset++] = uint4(
                cluster,
                reference_offset + first,
                min(uint(PISA_QUERY_TILE), count - first),
                0);
        }
        reference_offset += count;
    }
    atomic_store_explicit(tile_count, tile_offset, memory_order_relaxed);
    atomic_store_explicit(tile_count + 1, 1, memory_order_relaxed);
    atomic_store_explicit(tile_count + 2, 1, memory_order_relaxed);
}

kernel void fbt_pisa_full_scatter(
    device const uint* leaf_candidates [[buffer(0)]],
    device atomic_uint* cursors [[buffer(1)]],
    device uint* refs [[buffer(2)]],
    constant uint2& row_range [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= row_range.y * kPisaCandidates) return;
    const uint query = index / kPisaCandidates;
    const uint candidate_slot = index % kPisaCandidates;
    const uint row = row_range.x + query;
    const uint leaf = leaf_candidates[ulong(row) * kPisaCandidates + candidate_slot];
    const uint current = (row % kContext) / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    if (leaf == UINT_MAX || leaf >= current || leaf == 0 || leaf == previous) return;
    const uint sequence = row / kContext;
    const uint cluster = sequence * kLeaves + leaf;
    const uint offset = atomic_fetch_add_explicit(
        &cursors[cluster], 1, memory_order_relaxed);
    refs[offset] = (row << 4) | candidate_slot;
}

kernel void fbt_pisa_full_leaf_q4(
    device const half* qkv [[buffer(0)]],
    device const uint* refs [[buffer(1)]],
    device const uint4* tiles [[buffer(2)]],
    device const atomic_uint* tile_count [[buffer(3)]],
    device float* leaf_scores [[buffer(4)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint valid_tiles = atomic_load_explicit(tile_count, memory_order_relaxed);
    if (tile_index >= valid_tiles) return;
    const uint4 tile = tiles[tile_index];
    const uint cluster = tile.x;
    const bool active = simdgroup < tile.z;
    const uint sequence = cluster / kLeaves;
    const uint leaf = cluster % kLeaves;
    threadgroup half shared_key[kBlock * kHeadDim];
    for (uint index = tid; index < kBlock * kHeadDim; index += 128) {
        const uint token = index / kHeadDim;
        const uint dim = index % kHeadDim;
        const uint source = sequence * kContext + leaf * kBlock + token;
        shared_key[index] = qkv[ulong(source) * kQkvWidth + kQueryWidth + dim];
    }
    threadgroup float logits[PISA_QUERY_TILE][kQueryHeads][kBlock + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    const uint packed = refs[tile.y + simdgroup];
    const uint row = packed >> 4;
    const uint candidate_slot = packed & 15;
    const uint current = (row % kContext) / kBlock;
    const uint previous = current == 0 ? 0 : current - 1;
    const bool forced = leaf == 0 || leaf == previous || leaf == current;
    float grouped_score = forced ? INFINITY : -INFINITY;
    if (!forced && leaf < current) {
        simdgroup_half8x8 query[8];
        PISA_FRAGMENTS(dim, 8)
            simdgroup_load(
                query[dim / 8],
                qkv + ulong(row) * kQkvWidth + dim,
                kHeadDim,
                ulong2(0, 0));
        PISA_END_FRAGMENTS
        for (uint token = 0; token < kBlock; token += 8) {
            simdgroup_float8x8 score(0.0f);
            for (uint dim = 0; dim < kHeadDim; dim += 8) {
                simdgroup_half8x8 key;
                simdgroup_load(
                    key,
                    shared_key + token * kHeadDim + dim,
                    kHeadDim,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(
                    score,
                    query[dim / 8],
                    key,
                    score);
            }
            simdgroup_store(
                score,
                &logits[simdgroup][0][token],
                kBlock + 1,
                ulong2(0, 0));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        grouped_score = 0.0f;
        for (uint head = 0; head < kQueryHeads; ++head) {
            const float logit0 = logits[simdgroup][head][lane] * 0.125f;
            const float logit1 = logits[simdgroup][head][lane + 32] * 0.125f;
            const float maximum = simd_max(max(logit0, logit1));
            const float total = simd_sum(
                exp(logit0 - maximum) + exp(logit1 - maximum));
            grouped_score += maximum + log(total);
        }
    }
    if (lane == 0)
        leaf_scores[ulong(row) * kPisaCandidates + candidate_slot] = grouped_score;
}

kernel void fbt_pisa_full_final(
    device const uint* leaf_candidates [[buffer(0)]],
    device const float* leaf_scores [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    constant uint2& row_range [[buffer(3)]],
    uint query [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = row_range.x + query;
    if (query >= row_range.y || row >= kRows) return;
    const uint candidate = lane < kPisaCandidates
        ? leaf_candidates[ulong(row) * kPisaCandidates + lane]
        : UINT_MAX;
    float score = lane < kPisaCandidates && candidate != UINT_MAX
        ? leaf_scores[ulong(row) * kPisaCandidates + lane]
        : -INFINITY;
    uint selected = UINT_MAX;
    for (uint slot = 0; slot < kSelected; ++slot) {
        const float best_score = simd_max(score);
        const uint best = simd_min(
            score == best_score && score > -INFINITY ? candidate : UINT_MAX);
        if (lane == slot) selected = best;
        if (candidate == best) score = -INFINITY;
    }
    if (lane < kSelected)
        blocks[ulong(row) * kSelected + lane] = selected;
}
