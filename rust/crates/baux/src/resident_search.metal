#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef PROCEDURAL_COUNT
#define PROCEDURAL_COUNT 4
#endif

// BAUX_ZIGGURAT_TABLES

struct Leaf {
    ulong key;
    ulong offset;
    ulong length;
    float scale;
    float weight;
    uint address;
    uint pad;
};
struct Tile { uint leaf; uint start; uint length; uint pad; };
struct Params {
    ulong seed;
    ulong stream_seed;
    float radius;
    uint candidate;
    uint tiles;
    uint history;
    uint initialize;
    uint mode;
    uint base_slot;
    uint blocks;
};
struct SelectionParams {
    ulong root_seed;
    float outcomes[BAUX_HISTORY];
    float variances[BAUX_HISTORY];
    float draws[BAUX_HISTORY];
    float base_distances[BAUX_HISTORY];
    float local_scales[BAUX_HISTORY];
    float axis_history[BAUX_HISTORY];
    float axis_candidates[4];
    float axis_base;
    float axis_weight;
    uint axis_enabled;
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
    float radius;
    uint neighbors;
    uint history;
    uint acquisition;
    uint tiles;
    uint mode;
    uint resident_history;
    uint resident_indices[BAUX_HISTORY];
    uint implicit_history;
    uint forced_candidate;
    uint distance_metric;
    uint local_scale_neighbors;
    uint incumbent_index;
    uint candidate_floor;
    uint adaptive_neighbors;
    uint pad;
};
struct Decision {
    uint index;
    uint valid;
    ulong root_seed;
    ulong seed;
    float radius;
    float score;
    uint mode;
    uint pad;
    float predicted_mean;
    float predicted_standard_error;
    float incumbent_mean;
    float incumbent_standard_error;
    uint neighbors;
    uint incumbent_neighbors;
    float candidate_means[4];
    float candidate_standard_errors[4];
    float candidate_scores[4];
};
struct Partial { float anchor; float rejected; float squared; uint changed; uint invalid; };

inline ulong mix64(ulong x) {
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    return x ^ (x >> 31);
}
inline uint mix32(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
inline ulong2 ziggurat_domain(ulong seed, ulong key) {
    ulong lo = mix64(seed);
    ulong hi = mix64(key ^ 0xd6e8feb86659fd93ul);
    lo ^= mix64(hi ^ 0xa0761d6478bd642ful);
    hi ^= mix64(lo ^ 0xe7037ed1a0b428dbul);
    return ulong2(lo, hi);
}
inline float2 rademacher_pair(ulong2 domain, ulong pair) {
    ulong first = mix64(
        domain.x ^ mix64(domain.y + pair * 0x9e3779b97f4a7c15ul));
    return float2(
        (first & 1ul) == 0ul ? -1.0f : 1.0f,
        (first & 2ul) == 0ul ? -1.0f : 1.0f
    );
}
inline uint domain_bits(
    ulong2 domain,
    ulong element,
    uint attempt,
    uint stream) {
    ulong meta = (ulong(attempt) << 2) | ulong(stream);
    ulong word = mix64(
        domain.x ^ mix64(
            domain.y
            + element * 0x9e3779b97f4a7c15ul
            + meta * 0xd3a2646c9e3779b9ul));
    return uint(word ^ (word >> 32));
}
inline float ziggurat_uniform(uint bits) {
    return min(float((bits >> 8) + 1u) * 0x1.0p-24f, 0.99999994f);
}
#ifndef RADEMACHER_ONLY
inline float domain_sample(ulong2 domain, ulong element) {
    for (uint attempt = 0;; attempt++) {
        uint bits = domain_bits(domain, element, attempt, 0u);
        int signed_sample = as_type<int>(bits);
        uint layer = uint(signed_sample) & 255u;
        uint magnitude = signed_sample < 0
            ? (~uint(signed_sample) + 1u)
            : uint(signed_sample);
        float sample = float(signed_sample) * ziggurat_widths[layer];
        if (magnitude < ziggurat_thresholds[layer]) return sample;
        if (layer == 0u) {
            for (uint tail = 0;; tail++) {
                float x = -log(ziggurat_uniform(
                    domain_bits(domain, element, attempt + tail, 1u)))
                    / 3.654152885361009f;
                float y = -log(ziggurat_uniform(
                    domain_bits(domain, element, attempt + tail, 2u)));
                if (2.0f * y >= x * x) {
                    float value = 3.654152885361009f + x;
                    return signed_sample < 0 ? -value : value;
                }
            }
        }
        float uniform = ziggurat_uniform(
            domain_bits(domain, element, attempt, 1u));
        float density = ziggurat_densities[layer]
            + uniform * (ziggurat_densities[layer - 1u] - ziggurat_densities[layer]);
        if (density < exp(-0.5f * sample * sample)) return sample;
    }
}
inline float ziggurat_sample(ulong seed, ulong key, ulong element) {
    return domain_sample(ziggurat_domain(seed, key), element);
}
inline float2 domain_ziggurat(ulong2 domain, ulong pair) {
    return float2(
        domain_sample(domain, pair * 2ul),
        domain_sample(domain, pair * 2ul + 1ul));
}
#endif
#ifdef FP16_INDEPENDENT
inline float decode(ushort x) { return float(as_type<half>(x)); }
inline ushort encode(float x) { return as_type<ushort>(half(x)); }
inline float2 decode_pair(ushort2 x) { return float2(as_type<half2>(x)); }
inline ushort2 encode_pair(float2 x) { return as_type<ushort2>(half2(x)); }
inline bool invalid_value(ushort x) { return (x & 0x7c00u) == 0x7c00u; }
#else
inline float decode(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort encode(float x) {
    uint bits = as_type<uint>(x);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline bool invalid_value(ushort x) { return (x & 0x7f80u) == 0x7f80u; }
#endif
inline float2 domain_noise(ulong2 domain, ulong pair, uint mode) {
#ifdef RADEMACHER_ONLY
    return rademacher_pair(domain, pair);
#else
    if (mode == 1u) {
        return rademacher_pair(domain, pair);
    }
    return domain_ziggurat(domain, pair);
#endif
}
inline float2 noise_pair(ulong seed, ulong key, ulong pair, uint mode) {
    return domain_noise(ziggurat_domain(seed, key), pair, mode);
}
inline float sample(ulong seed, ulong key, ulong element, uint mode) {
    float2 pair = noise_pair(seed, key, element / 2, mode);
    return (element & 1ul) == 0ul ? pair.x : pair.y;
}
inline uint trial_hash(uint low, uint high, uint element) {
    uint value = low ^ (element * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= high;
    value *= 0x846ca68bu;
    return value ^ (value >> 15);
}
inline ulong candidate_seed(ulong root, uint candidate) {
#ifdef PROCEDURAL_SLOTS
    uint arm = candidate / PROCEDURAL_SLOTS;
    candidate %= PROCEDURAL_SLOTS;
    if (arm != 0u) {
        ulong z = (root ^ 0x70726f6361726d31ul ^ ulong(arm)) + 0x9e3779b97f4a7c15ul;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
        root = z ^ (z >> 31);
    }
#endif
    uint stream = candidate;
    uint low = uint(root);
    uint high = uint(root >> 32);
    return ulong(trial_hash(low, high, stream))
        | (ulong(trial_hash(high, low, stream ^ 0x9e3779b9u)) << 32);
}

inline float2 coord_step(
    float2 base_value,
    float step,
    float2 noise,
    ulong2 domain,
    ulong pair,
    uint coord) {
    if (coord == 0u) {
        return step * noise;
    }
    float norm = sqrt(max(noise.x * noise.x + noise.y * noise.y, 1.0e-24f));
    float2 unit_dir = noise / norm;
    if (coord == 1u) {
        return step * unit_dir;
    }
    if (coord == 2u) {
        float radial = ziggurat_uniform(domain_bits(domain, pair, 0u, 3u));
        return (step * radial) * unit_dir;
    }
    float r0_sq = base_value.x * base_value.x + base_value.y * base_value.y;
    if (r0_sq <= 1.0e-24f) {
        return step * unit_dir;
    }
    float r0 = sqrt(r0_sq);
    float2 raw = step * unit_dir;
    float proj = (raw.x * base_value.x + raw.y * base_value.y) / r0_sq;
    float2 tangent = raw - proj * base_value;
    if (tangent.x * tangent.x + tangent.y * tangent.y <= 1.0e-24f) {
        tangent = raw;
    }
    float2 disp = base_value + tangent;
    float disp_norm = sqrt(max(disp.x * disp.x + disp.y * disp.y, 1.0e-24f));
    return disp * (r0 / disp_norm) - base_value;
}

#ifdef FP16_INDEPENDENT
inline ulong proposal_seed(ulong root, uint candidate) {
    return candidate_seed(root, candidate);
}

inline float2 proposal_delta(
    float2 base_value,
    Leaf leaf,
    ulong2 domain,
    uint candidate,
    float radius,
    uint mode,
    ulong pair) {
    float2 noise = domain_noise(domain, pair, mode);
    return coord_step(base_value, leaf.scale * radius, noise, domain, pair, leaf.pad);
}

inline float2 continuous_pair(float2 base_value, Leaf leaf, ulong seed,
    uint candidate, float radius, uint mode,
    ulong pair, constant Params &p) {
    ulong2 domain = ziggurat_domain(seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
    return proposal_delta(base_value, leaf, domain, candidate, radius, mode, pair);
}

inline ushort2 domain_proposal(
    float2 base_value,
    Leaf leaf,
    ulong2 domain,
    uint candidate,
    float radius,
    uint mode,
    ulong pair) {
    return encode_pair(
        base_value + proposal_delta(base_value, leaf, domain, candidate, radius, mode, pair));
}

// Seed is the canonical candidate stream, before the single noise-domain salt.
// Every FP16 materialization path and sampled observation uses this operation.
inline ushort2 proposal_pair(float2 base_value, Leaf leaf, ulong seed,
    uint candidate, float radius, uint mode,
    ulong pair, constant Params &p) {
    ulong2 domain = ziggurat_domain(seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
    return domain_proposal(base_value, leaf, domain, candidate, radius, mode, pair);
}

inline ushort2 procedural_pair(float2 base_value, Leaf leaf, constant Params &p,
    ulong pair,
    uint candidate) {
    return proposal_pair(base_value, leaf,
        proposal_seed(p.stream_seed, candidate),
        candidate,
        p.radius,
        p.mode, pair, p);
}
#endif

kernel void dense_propose(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device const Tile *tiles [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    device Partial *partials [[buffer(6)]],
    device float *geometry [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float a = 0.0f, b = 0.0f, sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
#ifdef FP16_INDEPENDENT
        ulong first = (element / 2ul) * 2ul;
        ushort2 base_pair = ushort2(base[leaf.offset + first],
            first + 1ul < leaf.length ? base[leaf.offset + first + 1ul] : 0);
        ushort2 values = proposal_pair(decode_pair(base_pair), leaf, p.seed,
            p.candidate, p.radius, p.mode,
            element / 2ul, p);
        float base_value = decode(base[index]);
        ushort value = values[element & 1ul];
#else
        float base_value = decode(base[index]);
        ushort value;
        if (leaf.pad == 0u) {
            float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
            value = encode(base_value + (leaf.scale * p.radius) * noise);
        } else {
            ulong first = (element / 2ul) * 2ul;
            float2 base_pair = float2(decode(base[leaf.offset + first]),
                first + 1ul < leaf.length ? decode(base[leaf.offset + first + 1ul]) : 0.0f);
            ulong2 domain = ziggurat_domain(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
            float2 delta_pair = coord_step(base_pair, leaf.scale * p.radius,
                domain_noise(domain, element / 2ul, p.mode), domain, element / 2ul, leaf.pad);
            value = encode(base_value + delta_pair[element & 1ul]);
        }
#endif
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base[index]);
        float value_float = decode(value);
        float delta = value_float - base_value;
        sq += delta * delta;
        norm += (delta * delta) * leaf.weight;
        if (p.base_slot != 0) {
            delta = value_float - decode(anchor[index]);
            a += (delta * delta) * leaf.weight;
        }
        if (p.history == 2 && p.base_slot != 1) {
            delta = value_float - decode(rejected[index]);
            b += (delta * delta) * leaf.weight;
        }
    }
    if (p.base_slot == 0) a = norm;
    if (p.history == 2 && p.base_slot == 1) b = norm;
    threadgroup float sums_a[8], sums_b[8], sums_sq[8], sums_norm[8];
    threadgroup uint counts[8], flags[8];
    const uint lane = tid & 31u;
    const uint simdgroup = tid >> 5;
    const float a_sum = simd_sum(a);
    const float b_sum = simd_sum(b);
    const float sq_sum = simd_sum(sq);
    const float norm_sum = simd_sum(norm);
    const uint changed_sum = simd_sum(changed);
    const uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        sums_a[simdgroup] = a_sum;
        sums_b[simdgroup] = b_sum;
        sums_sq[simdgroup] = sq_sum;
        sums_norm[simdgroup] = norm_sum;
        counts[simdgroup] = changed_sum;
        flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const bool active = lane < 8;
        const uint slot = min(lane, 7u);
        const float total_a = simd_sum(active ? sums_a[slot] : 0.0f);
        const float total_b = simd_sum(active ? sums_b[slot] : 0.0f);
        const float total_sq = simd_sum(active ? sums_sq[slot] : 0.0f);
        const float total_norm = simd_sum(active ? sums_norm[slot] : 0.0f);
        const uint total_changed = simd_sum(active ? counts[slot] : 0u);
        const uint total_invalid = simd_sum(active ? flags[slot] : 0u);
        if (lane != 0) return;
        partials[p.candidate * p.tiles + tile_index] = {
            total_a, total_b, total_sq, total_changed, uint(total_invalid != 0)
        };
        geometry[p.candidate * p.tiles + tile_index] = total_norm;
    }
}

// First-round FP16 search has exactly one history row and the base is that row.
// Keeping a separate entry point removes two distance accumulators and all
// history/reference traffic from the billion-coordinate hot loop.
kernel void dense_propose_initial(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device const Tile *tiles [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    device Partial *partials [[buffer(6)]],
    device float *geometry [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
#ifdef FP16_INDEPENDENT
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + index)
            : ushort2(base[index], complete_pair ? base[index + 1] : 0);
        float2 base_values = decode_pair(base_pair);
        ushort2 values = proposal_pair(base_values, leaf, p.seed,
            p.candidate, p.radius, p.mode, element / 2ul, p);
        if (aligned_pair) {
            *reinterpret_cast<device ushort2*>(output + index) = values;
        } else {
            output[index] = values.x;
            if (complete_pair) output[index + 1] = values.y;
        }
        uint pair_length = complete_pair ? 2u : 1u;
        for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
            ushort value = values[pair_offset];
            invalid |= uint(invalid_value(value));
            changed += uint(value != base_pair[pair_offset]);
            float delta = decode(value) - base_values[pair_offset];
            sq += delta * delta;
            norm += (delta * delta) * leaf.weight;
        }
    }
#else
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        ushort base_bits = base[index];
        float base_value = decode(base_bits);
        ushort value;
        if (leaf.pad == 0u) {
            float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
            value = encode(base_value + (leaf.scale * p.radius) * noise);
        } else {
            ulong first = (element / 2ul) * 2ul;
            float2 base_pair = float2(decode(base[leaf.offset + first]),
                first + 1ul < leaf.length ? decode(base[leaf.offset + first + 1ul]) : 0.0f);
            ulong2 domain = ziggurat_domain(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
            float2 delta_pair = coord_step(base_pair, leaf.scale * p.radius,
                domain_noise(domain, element / 2ul, p.mode), domain, element / 2ul, leaf.pad);
            value = encode(base_value + delta_pair[element & 1ul]);
        }
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base_bits);
        float delta = decode(value) - base_value;
        sq += delta * delta;
        norm += (delta * delta) * leaf.weight;
    }
#endif
    threadgroup float sums_sq[8], sums_norm[8];
    threadgroup uint counts[8], flags[8];
    const uint lane = tid & 31u;
    const uint simdgroup = tid >> 5;
    const float sq_sum = simd_sum(sq);
    const float norm_sum = simd_sum(norm);
    const uint changed_sum = simd_sum(changed);
    const uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        sums_sq[simdgroup] = sq_sum;
        sums_norm[simdgroup] = norm_sum;
        counts[simdgroup] = changed_sum;
        flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const bool active = lane < 8;
        const uint slot = min(lane, 7u);
        const float total_sq = simd_sum(active ? sums_sq[slot] : 0.0f);
        const float total_norm = simd_sum(active ? sums_norm[slot] : 0.0f);
        const uint total_changed = simd_sum(active ? counts[slot] : 0u);
        const uint total_invalid = simd_sum(active ? flags[slot] : 0u);
        if (lane != 0) return;
        partials[p.candidate * p.tiles + tile_index] = {
            total_norm, 0.0f, total_sq, total_changed, uint(total_invalid != 0)
        };
        geometry[p.candidate * p.tiles + tile_index] = total_norm;
    }
}


kernel void dense_propose_pool(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device const Tile *tiles [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    device Partial *partials [[buffer(6)]],
    device float *geometry [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    float dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#endif
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
#ifdef FP16_INDEPENDENT
        ulong first_index = leaf.offset + pair_element;
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (first_index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + first_index)
            : ushort2(base[first_index], complete_pair ? base[first_index + 1] : 0);
        float2 base_values = decode_pair(base_pair);
        float2 anchor_values = base_values;
        if (p.base_slot != 0) {
            ushort2 anchor_pair = aligned_pair
                ? *reinterpret_cast<device const ushort2*>(anchor + first_index)
                : ushort2(anchor[first_index], complete_pair ? anchor[first_index + 1] : 0);
            anchor_values = decode_pair(anchor_pair);
        }
        float2 rejected_values = float2(0.0f);
        if (p.history == 2 && p.base_slot != 1) {
            ushort2 rejected_pair = aligned_pair
                ? *reinterpret_cast<device const ushort2*>(rejected + first_index)
                : ushort2(rejected[first_index], complete_pair ? rejected[first_index + 1] : 0);
            rejected_values = decode_pair(rejected_pair);
        } else if (p.history == 2) {
            rejected_values = base_values;
        }
        ushort2 values[4];
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++)
            values[candidate] = procedural_pair(
                base_values,
                leaf,
                p,
                pair,
                candidate);
#if PROCEDURAL_COUNT == 1
        if (p.initialize == 0) {
            if (aligned_pair) {
                *reinterpret_cast<device ushort2*>(output + first_index) = values[0];
            } else {
                output[first_index] = values[0].x;
                if (complete_pair) output[first_index + 1] = values[0].y;
            }
        }
#endif
        uint pair_length = complete_pair ? 2u : 1u;
        for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
            ushort base_bits = base_pair[pair_offset];
            float base_value = base_values[pair_offset];
            float anchor_value = anchor_values[pair_offset];
            float rejected_value = rejected_values[pair_offset];
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                ushort value = values[candidate][pair_offset];
                invalid[candidate] |= uint(invalid_value(value));
                changed[candidate] += uint(value != base_bits);
                float value_float = decode(value);
                float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                deltas[candidate] = delta;
#endif
                sq[candidate] += delta * delta;
                norms[candidate] += (delta * delta) * leaf.weight;
                if (p.base_slot != 0) {
                    delta = value_float - anchor_value;
                    a[candidate] += (delta * delta) * leaf.weight;
                }
                if (p.history == 2 && p.base_slot != 1) {
                    delta = value_float - rejected_value;
                    b[candidate] += (delta * delta) * leaf.weight;
                }
            }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            dots[0] += deltas[0] * deltas[1] * leaf.weight;
            dots[1] += deltas[0] * deltas[2] * leaf.weight;
            dots[2] += deltas[0] * deltas[3] * leaf.weight;
            dots[3] += deltas[1] * deltas[2] * leaf.weight;
            dots[4] += deltas[1] * deltas[3] * leaf.weight;
            dots[5] += deltas[2] * deltas[3] * leaf.weight;
#endif
        }
#else
        for (uint pair_offset = 0; pair_offset < 2; pair_offset++) {
            uint item_offset = item + pair_offset;
            if (item_offset >= tile.length) break;
            ulong element = ulong(tile.start) + item_offset;
            ulong index = leaf.offset + element;
            ushort base_bits = base[index];
            float base_value = decode(base_bits);
            float anchor_value = base_value;
            if (p.base_slot != 0) anchor_value = decode(anchor[index]);
            float rejected_value = 0.0f;
            if (p.history == 2 && p.base_slot != 1)
                rejected_value = decode(rejected[index]);
            else if (p.history == 2)
                rejected_value = base_value;
            float radius = leaf.scale * p.radius;
            ushort values[4];
            ulong first = (element / 2ul) * 2ul;
            float2 base_pair = float2(decode(base[leaf.offset + first]),
                first + 1ul < leaf.length ? decode(base[leaf.offset + first + 1ul]) : 0.0f);
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                ulong cseed = candidate_seed(p.stream_seed, candidate) ^ 0x8ebc6af09c88c6e3ul;
                ulong2 domain = ziggurat_domain(cseed, leaf.key);
                float2 samples = domain_noise(domain, pair, p.mode);
                float2 delta_pair = coord_step(base_pair, radius, samples, domain, pair, leaf.pad);
                values[candidate] = encode(base_value + delta_pair[pair_offset]);
            }
#if PROCEDURAL_COUNT == 1
            if (p.initialize == 0) {
                output[index] = values[0];
            }
#endif
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                ushort value = values[candidate];
                invalid[candidate] |= uint(invalid_value(value));
                changed[candidate] += uint(value != base_bits);
                float value_float = decode(value);
                float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                deltas[candidate] = delta;
#endif
                sq[candidate] += delta * delta;
                norms[candidate] += (delta * delta) * leaf.weight;
                if (p.base_slot != 0) {
                    delta = value_float - anchor_value;
                    a[candidate] += (delta * delta) * leaf.weight;
                }
                if (p.history == 2 && p.base_slot != 1) {
                    delta = value_float - rejected_value;
                    b[candidate] += (delta * delta) * leaf.weight;
                }
            }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            dots[0] += deltas[0] * deltas[1] * leaf.weight;
            dots[1] += deltas[0] * deltas[2] * leaf.weight;
            dots[2] += deltas[0] * deltas[3] * leaf.weight;
            dots[3] += deltas[1] * deltas[2] * leaf.weight;
            dots[4] += deltas[1] * deltas[3] * leaf.weight;
            dots[5] += deltas[2] * deltas[3] * leaf.weight;
#endif
        }
#endif
    }
    if (p.base_slot == 0) {
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) a[candidate] = norms[candidate];
    } else if (p.base_slot == 1) {
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) b[candidate] = norms[candidate];
    }
    threadgroup float reduced_a[32], reduced_b[32], reduced_sq[32], reduced_norm[32];
    threadgroup uint reduced_counts[32], reduced_flags[32];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        float a_sum = simd_sum(a[candidate]);
        float b_sum = simd_sum(b[candidate]);
        float sq_sum = simd_sum(sq[candidate]);
        float norm_sum = simd_sum(norms[candidate]);
        uint changed_sum = simd_sum(changed[candidate]);
        uint invalid_sum = simd_sum(invalid[candidate]);
        if (lane == 0) {
            uint slot = candidate * 8 + simdgroup;
            reduced_a[slot] = a_sum;
            reduced_b[slot] = b_sum;
            reduced_sq[slot] = sq_sum;
            reduced_norm[slot] = norm_sum;
            reduced_counts[slot] = changed_sum;
            reduced_flags[slot] = invalid_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
            bool active = lane < 8;
            uint slot = candidate * 8 + min(lane, 7u);
            float a_sum = simd_sum(active ? reduced_a[slot] : 0.0f);
            float b_sum = simd_sum(active ? reduced_b[slot] : 0.0f);
            float sq_sum = simd_sum(active ? reduced_sq[slot] : 0.0f);
            float norm_sum = simd_sum(active ? reduced_norm[slot] : 0.0f);
            uint changed_sum = simd_sum(active ? reduced_counts[slot] : 0u);
            uint invalid_sum = simd_sum(active ? reduced_flags[slot] : 0u);
            if (lane != 0) continue;
            partials[(p.initialize * 4 + candidate) * p.tiles + tile_index] =
                {a_sum, b_sum, sq_sum, changed_sum, uint(invalid_sum != 0)};
#ifndef POOL_GEOMETRY_DIAGNOSTICS
            if (p.initialize == 0)
                geometry[candidate * p.tiles + tile_index] = norm_sum;
#endif
        }
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float sums_a[1024], sums_b[1024], sums_sq[512];
    for (uint candidate = 0; candidate < 4; candidate++)
        sums_a[candidate * 256 + tid] = norms[candidate];
    for (uint pair = 0; pair < 4; pair++)
        sums_b[pair * 256 + tid] = dots[pair];
    for (uint pair = 0; pair < 2; pair++)
        sums_sq[pair * 256 + tid] = dots[pair + 4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint candidate = 0; candidate < 4; candidate++) {
                uint slot = candidate * 256 + tid;
                sums_a[slot] += sums_a[slot + stride];
            }
            for (uint pair = 0; pair < 4; pair++) {
                uint slot = pair * 256 + tid;
                sums_b[slot] += sums_b[slot + stride];
            }
            for (uint pair = 0; pair < 2; pair++) {
                uint slot = pair * 256 + tid;
                sums_sq[slot] += sums_sq[slot + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && p.initialize == 0) {
        for (uint candidate = 0; candidate < 4; candidate++)
            geometry[candidate * p.tiles + tile_index] = sums_a[candidate * 256];
        for (uint pair = 0; pair < 4; pair++)
            geometry[(pair + 4) * p.tiles + tile_index] = sums_b[pair * 256];
        for (uint pair = 0; pair < 2; pair++)
            geometry[(pair + 8) * p.tiles + tile_index] = sums_sq[pair * 256];
    }
#endif
}

inline float acquisition_score(
    thread const float *distances,
    constant SelectionParams &q,
    uint excluded_index,
    thread float &mean_out,
    thread float &standard_error_out,
    thread uint &neighbors_out) {
    float scaled_distances[BAUX_HISTORY];
    if (q.distance_metric != 0) {
        float query_neighbors[BAUX_HISTORY];
        uint query_count = 0;
        for (uint i = 0; i < q.history; i++) {
            if (i == excluded_index || distances[i] <= 0.0f) continue;
            uint position = query_count;
            while (position > 0 && distances[i] < query_neighbors[position - 1]) {
                query_neighbors[position] = query_neighbors[position - 1];
                position--;
            }
            query_neighbors[position] = distances[i];
            query_count++;
        }
        uint local_rank = min(q.local_scale_neighbors, query_count);
        float query_scale = local_rank == 0
            ? 1.0f
            : max(query_neighbors[local_rank - 1], 1.0e-12f);
        for (uint i = 0; i < q.history; i++) {
            float s_q = sqrt(query_scale), s_h = sqrt(max(q.local_scales[i], 1.0e-12f));
            if (q.distance_metric == 3) {
                float diff = abs(s_q - s_h);
                scaled_distances[i] = distances[i] / sqrt(max(s_q * s_h, 1.0e-12f)) + diff * diff;
            } else {
                float hyp = float(q.distance_metric >> 1);
                scaled_distances[i] = (distances[i] + hyp * (s_q - s_h) * (s_q - s_h)) / ((1.0f + hyp) * s_q * s_h);
            }
        }
    } else {
        for (uint i = 0; i < q.history; i++) scaled_distances[i] = distances[i];
    }
    uint nearest[BAUX_HISTORY];
    uint available = 0;
    for (uint i = 0; i < q.history; i++) {
        if (i == excluded_index) continue;
        uint position = available++;
        while (position > 0
            && scaled_distances[i] < scaled_distances[nearest[position - 1]]) {
            nearest[position] = nearest[position - 1];
            position--;
        }
        nearest[position] = i;
    }
    uint count = min(q.neighbors, available);
    if (count == 0) {
        mean_out = q.outcomes[q.incumbent_index];
        standard_error_out = sqrt(max(q.aleatoric_scale, 1.0e-12f)) * q.y_scale;
        neighbors_out = 0;
        return mean_out;
    }
    float y_scale_sq = max(q.y_scale * q.y_scale, 1.0e-12f);
    if (q.adaptive_neighbors != 0) {
        float precision_sum = 0.0f, noise_sum = 0.0f, best_risk = INFINITY;
        uint chosen = 1;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float noise = q.aleatoric_scale + q.variances[index] / y_scale_sq;
            float precision = 1.0f / max(1.0e-9f + q.epistemic_scale * scaled_distances[index] + noise, 1.0e-12f);
            precision_sum += precision;
            noise_sum += precision * precision * noise;
            float variance_error = noise_sum / max(precision_sum * precision_sum, 1.0e-24f);
            float risk = q.epistemic_scale * scaled_distances[index] + variance_error;
            if (risk < best_risk) { best_risk = risk; chosen = position + 1; }
        }
        count = chosen;
    }
    neighbors_out = count;
    float sum = 0.0f;
    float value = 0.0f;
    float reference = 1.17549435e-38f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + q.variances[index] / y_scale_sq;
        float weight = 1.0f / max(variance, 1.0e-12f);
        sum += weight;
        value += weight * q.outcomes[index];
        reference = max(reference, weight);
    }
    float mean = value / max(sum, 1.0e-12f);
    float aleatoric = 0.0f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + q.variances[index] / y_scale_sq;
        float weight = (1.0f / max(variance, 1.0e-12f)) / max(sum, 1.0e-12f);
        aleatoric += weight * (q.aleatoric_scale + q.variances[index] / y_scale_sq);
    }
    float epistemic_var = 1.0f / max(sum, 1.0e-12f);
    float standard_error = sqrt(epistemic_var + aleatoric) * q.y_scale;
    float epistemic_se = sqrt(epistemic_var) * q.y_scale;
    mean_out = mean;
    standard_error_out = standard_error;
    if (q.acquisition == 1) {
        float noise = 0.0f;
        float squared = 0.0f;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float variance = 1.0e-9f
                + q.epistemic_scale * scaled_distances[index]
                + q.aleatoric_scale
                + q.variances[index] / y_scale_sq;
            float weight = (1.0f / max(variance, 1.0e-12f)) / reference;
            noise += weight * q.draws[index];
            squared += weight * weight;
        }
        return mean + epistemic_se * (noise / max(sqrt(squared), 1.0e-12f));
    }
    if (q.acquisition == 2) return mean + epistemic_se;
    return mean + q.beta * epistemic_se;
}

#ifdef POOL_GEOMETRY_DIAGNOSTICS
// Invoke the production acquisition operation directly; do not duplicate it
// in the numerical parity harness.
kernel void dense_posterior_probe(
    constant SelectionParams &q [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &excluded [[buffer(3)]]) {
    float distances[BAUX_HISTORY];
    for (uint i = 0; i < q.history; ++i) distances[i] = input[i];
    float mean, error;
    uint neighbors;
    output[0] = acquisition_score(distances, q, excluded, mean, error, neighbors);
    output[1] = mean;
    output[2] = error;
    output[3] = float(neighbors);
}
#endif

// Collapse the per-tile records in parallel before the deliberately scalar
// acquisition step.  The old selector walked every tile on one GPU thread.
kernel void dense_reduce_pool(
    device const Partial *partials [[buffer(0)]],
    device const float *geometry [[buffer(1)]],
    device Partial *aggregates [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    float anchor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rejected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    for (uint tile = tid; tile < q.tiles; tile += 256) {
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
            Partial p = partials[(pair * 4 + candidate) * q.tiles + tile];
            anchor[candidate] += p.anchor;
            rejected[candidate] += p.rejected;
            changed[candidate] |= p.changed;
            invalid[candidate] |= p.invalid | uint(!isfinite(p.squared));
            if (pair == 0)
                norm[candidate] += geometry[candidate * q.tiles + tile];
        }
    }
    threadgroup float anchor_sums[1024], rejected_sums[1024], norm_sums[1024];
    threadgroup uint changed_flags[1024], invalid_flags[1024];
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        uint slot = candidate * 256 + tid;
        anchor_sums[slot] = anchor[candidate];
        rejected_sums[slot] = rejected[candidate];
        norm_sums[slot] = norm[candidate];
        changed_flags[slot] = changed[candidate];
        invalid_flags[slot] = invalid[candidate];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                uint slot = candidate * 256 + tid;
                uint other = slot + stride;
                anchor_sums[slot] += anchor_sums[other];
                rejected_sums[slot] += rejected_sums[other];
                norm_sums[slot] += norm_sums[other];
                changed_flags[slot] |= changed_flags[other];
                invalid_flags[slot] |= invalid_flags[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint candidate = 0; candidate < 4; candidate++) {
            if (candidate < PROCEDURAL_COUNT) {
                uint slot = candidate * 256;
                aggregates[pair * 4 + candidate] = {
                    anchor_sums[slot],
                    rejected_sums[slot],
                    norm_sums[slot],
                    changed_flags[slot],
                    invalid_flags[slot]
                };
            } else {
                aggregates[pair * 4 + candidate] = {0.0f, 0.0f, 0.0f, 0u, 0u};
            }
        }
    }
}

kernel void dense_select(
    device const Partial *aggregates [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device float *pool_distances [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    float distances[4][BAUX_HISTORY];
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool changed[4] = {false, false, false, false};
    bool invalid[4] = {false, false, false, false};
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        proposal_norms[candidate] = aggregates[candidate].squared;
        for (uint row = 0; row < q.history; row++) {
            distances[candidate][row] = q.implicit_history != 0
                ? q.base_distances[row] + proposal_norms[candidate]
                : 0.0f;
        }
        for (uint row = 0; row < q.resident_history; row += 2) {
            Partial aggregate = aggregates[(row / 2) * 4 + candidate];
            float first = aggregate.anchor;
            float second = aggregate.rejected;
            changed[candidate] = changed[candidate] || aggregate.changed != 0;
            invalid[candidate] = invalid[candidate] || aggregate.invalid != 0;
            uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
            if (first_index < q.history) {
                distances[candidate][first_index] = first;
            }
            if (row + 1 < q.resident_history) {
                uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
                if (second_index < q.history) {
                    distances[candidate][second_index] = second;
                }
            }
            invalid[candidate] = invalid[candidate] || !isfinite(first)
                || (row + 1 < q.resident_history && !isfinite(second));
        }
        if (q.axis_enabled != 0) {
            for (uint row = 0; row < q.history; row++) {
                float delta = q.axis_candidates[candidate] - q.axis_history[row];
                distances[candidate][row] += q.axis_weight * delta * delta;
            }
        }
    }
    bool any_changed = false;
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        changed[candidate] = changed[candidate] || (q.axis_enabled != 0
            && q.axis_candidates[candidate] != q.axis_base);
        any_changed = any_changed || changed[candidate];
    }
    uint selected = 4;
    float best = -INFINITY;
    float candidate_means[4];
    float candidate_standard_errors[4];
    float candidate_scores[4];
    uint candidate_neighbors[4];
    for (uint candidate = 0; candidate < 4; candidate++) {
        candidate_means[candidate] = NAN;
        candidate_standard_errors[candidate] = NAN;
        candidate_scores[candidate] = NAN;
    }
    float incumbent_distances[BAUX_HISTORY];
    for (uint row = 0; row < q.history; row++)
        incumbent_distances[row] = q.base_distances[row]
            + (q.axis_enabled != 0
                ? q.axis_weight * (q.axis_base - q.axis_history[row])
                    * (q.axis_base - q.axis_history[row])
                : 0.0f);
    float incumbent_mean;
    float incumbent_standard_error;
    uint incumbent_neighbors;
    acquisition_score(
        incumbent_distances,
        q,
        BAUX_HISTORY,
        incumbent_mean,
        incumbent_standard_error,
        incumbent_neighbors);
    bool eligible[4] = {false, false, false, false};
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        if (candidate < q.candidate_floor
            || invalid[candidate]
            || (any_changed && !changed[candidate])) continue;
        float mean;
        float standard_error;
        uint neighbors;
        float score = acquisition_score(distances[candidate], q, BAUX_HISTORY, mean, standard_error, neighbors);
        candidate_means[candidate] = mean;
        candidate_standard_errors[candidate] = standard_error;
        candidate_scores[candidate] = score;
        candidate_neighbors[candidate] = neighbors;
        eligible[candidate] = isfinite(score) && isfinite(mean) && isfinite(standard_error);
        if (q.forced_candidate < 4) {
            if (candidate == q.forced_candidate) {
                selected = candidate;
                best = 0.0f;
            }
            continue;
        }
        if (q.acquisition != 2 && isfinite(score) && (selected == 4 || score > best)) {
            selected = candidate;
            best = score;
        }
    }
    if (q.acquisition == 2 && q.forced_candidate >= 4) {
        ulong best_priority = 0ul;
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
            if (!eligible[candidate]) continue;
            bool dominated = false;
            for (uint other = 0; other < PROCEDURAL_COUNT; other++) {
                if (other == candidate || !eligible[other]) continue;
                bool no_worse = candidate_means[other] >= candidate_means[candidate]
                    && candidate_standard_errors[other] >= candidate_standard_errors[candidate];
                bool strictly_better = candidate_means[other] > candidate_means[candidate]
                    || candidate_standard_errors[other] > candidate_standard_errors[candidate];
                dominated = dominated || (no_worse && strictly_better);
            }
            if (!dominated) {
                ulong priority = mix64(q.root_seed ^ (ulong(candidate) * 0x9e3779b97f4a7c15ul));
                if (selected == 4 || priority > best_priority) {
                    selected = candidate;
                    best_priority = priority;
                    best = candidate_scores[candidate];
                }
            }
        }
    }
    decision->index = selected;
    decision->valid = selected < 4 ? 1u : 0u;
    decision->root_seed = q.root_seed;
    decision->seed = selected < 4 ? candidate_seed(q.root_seed, selected) : 0ul;
    decision->radius = selected < 4 ? q.radius : 0.0f;
    decision->score = selected < 4 ? best : 0.0f;
    decision->mode = q.mode;
    decision->predicted_mean = selected < 4 ? candidate_means[selected] : 0.0f;
    decision->predicted_standard_error = selected < 4
        ? candidate_standard_errors[selected]
        : 0.0f;
    decision->incumbent_mean = incumbent_mean;
    decision->incumbent_standard_error = incumbent_standard_error;
    decision->neighbors = selected < 4 ? candidate_neighbors[selected] : 0;
    decision->incumbent_neighbors = incumbent_neighbors;
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        decision->candidate_means[candidate] = candidate_means[candidate];
        decision->candidate_standard_errors[candidate] = candidate_standard_errors[candidate];
        decision->candidate_scores[candidate] = candidate_scores[candidate];
    }
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++)
        for (uint row = 0; row < q.history; row++)
            pool_distances[candidate * q.history + row] = distances[candidate][row];
}

kernel void dense_materialize(
    device const ushort *base [[buffer(0)]],
    device const Leaf *leaves [[buffer(1)]],
    device const Tile *tiles [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    device const Decision *decision [[buffer(4)]],
    constant uint &account [[buffer(5)]],
    device Partial *partials [[buffer(6)]],
    constant Params &p [[buffer(7)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (decision->valid == 0) return;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
#ifdef FP16_INDEPENDENT
    ulong2 domain = ziggurat_domain(decision->seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
#else
    ulong seed = decision->seed ^ 0x8ebc6af09c88c6e3ul;
    const float step = leaf.scale * decision->radius;
#endif
    float squared = 0.0f;
    uint changed = 0u;
    uint invalid = 0u;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
#ifdef FP16_INDEPENDENT
        bool complete_pair = item + 1 < tile.length;
        bool aligned_pair = complete_pair && (index & 1ul) == 0ul;
        ushort2 base_pair = aligned_pair
            ? *reinterpret_cast<device const ushort2*>(base + index)
            : ushort2(base[index], complete_pair ? base[index + 1] : 0);
        ushort2 result = domain_proposal(decode_pair(base_pair), leaf, domain,
            decision->index,
            decision->radius, decision->mode, pair);
        if (aligned_pair) {
            *reinterpret_cast<device ushort2*>(output + index) = result;
        } else {
            output[index] = result.x;
            if (complete_pair) output[index + 1] = result.y;
        }
        if (account != 0u) {
            uint pair_length = complete_pair ? 2u : 1u;
            for (uint pair_offset = 0; pair_offset < pair_length; pair_offset++) {
                ushort value = result[pair_offset];
                changed += uint(value != base_pair[pair_offset]);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - decode(base_pair[pair_offset]);
                squared += delta * delta;
            }
        }
#else
        ulong2 domain = ziggurat_domain(seed, leaf.key);
        float2 samples = domain_noise(domain, pair, decision->mode);
        ushort first_bits = base[index];
        float2 base_pair = float2(decode(first_bits),
            item + 1 < tile.length ? decode(base[index + 1]) : 0.0f);
        float2 delta_pair = coord_step(base_pair, step, samples, domain, pair, leaf.pad);

        output[index] = encode(base_pair.x + delta_pair.x);
        if (account != 0u) {
            ushort value = output[index];
            changed += uint(value != first_bits);
            invalid |= uint(invalid_value(value));
            float delta = decode(value) - base_pair.x;
            squared += delta * delta;
        }

        if (item + 1 < tile.length) {
            ushort second_bits = base[index + 1];
            output[index + 1] = encode(base_pair.y + delta_pair.y);
            if (account != 0u) {
                ushort value = output[index + 1];
                changed += uint(value != second_bits);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - base_pair.y;
                squared += delta * delta;
            }
        }
#endif
    }
    if (account == 0u) return;
    threadgroup float squared_sums[8];
    threadgroup uint changed_sums[8], invalid_flags[8];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    float squared_sum = simd_sum(squared);
    uint changed_sum = simd_sum(changed);
    uint invalid_sum = simd_sum(invalid);
    if (lane == 0) {
        squared_sums[simdgroup] = squared_sum;
        changed_sums[simdgroup] = changed_sum;
        invalid_flags[simdgroup] = invalid_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        bool active = lane < 8;
        uint slot = min(lane, 7u);
        float total_squared = simd_sum(active ? squared_sums[slot] : 0.0f);
        uint total_changed = simd_sum(active ? changed_sums[slot] : 0u);
        uint total_invalid = simd_sum(active ? invalid_flags[slot] : 0u);
        if (lane == 0)
            partials[decision->index * p.tiles + tile_index] = {
                0.0f, 0.0f, total_squared, total_changed,
                uint(total_invalid != 0u)
            };
    }
}
