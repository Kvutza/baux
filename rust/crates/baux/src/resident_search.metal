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
    ulong basis_seed;
    float radius;
    float alternate_radius;
    uint candidate;
    uint tiles;
    uint history;
    uint initialize;
    uint mode;
    uint base_slot;
    uint blocks;
    uint program;
};
struct ReplayStep {
    ulong seed;
    float radius;
    uint accepted;
    uint candidate;
    uint mode;
};
struct SelectionParams {
    ulong root_seed;
    ulong basis_seed;
    float outcomes[BAUX_HISTORY];
    float variances[BAUX_HISTORY];
    float draws[BAUX_HISTORY];
    float base_distances[BAUX_HISTORY];
    float local_scales[BAUX_HISTORY];
    float latent_norms[4];
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
    float alternate_radius;
    uint neighbors;
    uint history;
    uint acquisition;
    uint tiles;
    uint mode;
    uint program;
    uint resident_history;
    uint resident_indices[2];
    uint implicit_history;
    uint exact_history;
    uint latent_history;
    uint forced_candidate;
    uint distance_scaling;
    uint local_scale_neighbors;
    uint incumbent_index;
    uint candidate_floor;
    uint sampled_history;
    uint adaptive_neighbors;
};
struct Decision {
    uint index;
    uint valid;
    ulong root_seed;
    ulong seed;
    ulong basis_seed;
    float radius;
    float score;
    uint mode;
    uint program;
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
#ifndef RADEMACHER_ONLY
inline uint ziggurat_bits_domain(
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
inline float ziggurat_sample_domain(ulong2 domain, ulong element) {
    for (uint attempt = 0;; attempt++) {
        uint bits = ziggurat_bits_domain(domain, element, attempt, 0u);
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
                    ziggurat_bits_domain(domain, element, attempt + tail, 1u)))
                    / 3.654152885361009f;
                float y = -log(ziggurat_uniform(
                    ziggurat_bits_domain(domain, element, attempt + tail, 2u)));
                if (2.0f * y >= x * x) {
                    float value = 3.654152885361009f + x;
                    return signed_sample < 0 ? -value : value;
                }
            }
        }
        float uniform = ziggurat_uniform(
            ziggurat_bits_domain(domain, element, attempt, 1u));
        float density = ziggurat_densities[layer]
            + uniform * (ziggurat_densities[layer - 1u] - ziggurat_densities[layer]);
        if (density < exp(-0.5f * sample * sample)) return sample;
    }
}
inline float ziggurat_sample(ulong seed, ulong key, ulong element) {
    return ziggurat_sample_domain(ziggurat_domain(seed, key), element);
}
inline float2 ziggurat_pair_domain(ulong2 domain, ulong pair) {
    return float2(
        ziggurat_sample_domain(domain, pair * 2ul),
        ziggurat_sample_domain(domain, pair * 2ul + 1ul));
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
inline float2 noise_pair(ulong seed, ulong key, ulong pair, uint mode) {
    ulong2 domain = ziggurat_domain(seed, key);
#ifdef RADEMACHER_ONLY
    return rademacher_pair(domain, pair);
#else
    if (mode == 1u) {
        return rademacher_pair(domain, pair);
    }
    return ziggurat_pair_domain(domain, pair);
#endif
}
inline float sample(ulong seed, ulong key, ulong element, uint mode) {
    float2 pair = noise_pair(seed, key, element / 2, mode);
    return (element & 1ul) == 0ul ? pair.x : pair.y;
}
inline float direction(float reference, float inverse_rms, float noise, uint candidate) {
#if defined(FP16_INDEPENDENT) || (defined(PROCEDURAL_SLOTS) && PROCEDURAL_SLOTS == 1)
    return noise;
#else
    return candidate < 2 ? 0.75f * (reference * inverse_rms) + sqrt(0.4375f) * noise : noise;
#endif
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
    uint stream = candidate / 2;
    uint low = uint(root);
    uint high = uint(root >> 32);
    return ulong(trial_hash(low, high, stream))
        | (ulong(trial_hash(high, low, stream ^ 0x9e3779b9u)) << 32);
}

inline float proposal_radius(
    uint candidate,
    uint program,
    float radius,
    float alternate_radius) {
#ifdef PROCEDURAL_SLOTS
    uint slot = program == 3u ? candidate : candidate % PROCEDURAL_SLOTS;
#else
    uint slot = candidate;
#endif
    return (slot & 1u) == 0u ? radius : alternate_radius;
}

#ifdef FP16_INDEPENDENT
inline ulong proposal_seed(ulong root, uint candidate, uint program) {
    return candidate_seed(root, candidate);
}

// Layerwise symmetry-compatible perturbation operator (arXiv:2605.18106):
// Applies exact linear variance-preserving quotient projections matched to each
// module's forward-pass symmetry group encoded in `leaf.address & 31u`:
// - `4u` (embedding_readout [512, 262144]): on the 64 tied hierarchical octal-tree
//   readout rows (d < 64), decomposes each 8-leaf sibling block into orthogonal
//   Level-5 softmax quotient (\Pi_8^\perp z) and Levels 0..4 tree-centroid (\mu_8 1_8)
//   components with variance-preserving weights (\alpha = 1.05220856, \beta = 0.50,
//   satisfying (7/8)\alpha^2 + (1/8)\beta^2 = 1) that concentrate energy onto the
//   Level-5 horizontal quotient while damping coarse centroid drift; on input-only
//   token rows (d >= 64), uses 1-pair isometric draws.
// - `0u` (router [512, 640]), `5u, 7u, 9u` (RMSNorms [512]), `>= 10u` (mHC 4-stream
//   Sinkhorn predictors/biases): exact 4-element horizontal zero-sum quotient
//   projection \Pi_4^\perp z = \sqrt{4/3} (z - \mu_4 1_4).
// - `1u, 2u, 3u, 6u` (linear projections and MoE experts): 1-pair isometric draw.
inline float2 sym_pair_continuous(
    float2 base_value,
    Leaf leaf,
    ulong salted_seed,
    float radius,
    uint mode,
    ulong pair) {
    float step = leaf.scale * radius;
    if (step == 0.0f) return float2(0.0f, 0.0f);
    uint family_code = leaf.address & 31u;
    ulong leaf_pairs = (leaf.length + 1ul) / 2ul;
    if (family_code == 4u && (pair >> 17) < 64ul && leaf_pairs >= 4ul) {
        ulong pair_base = min(pair & ~3ul, leaf_pairs - 4ul);
        uint sub = uint(pair - pair_base);
        float2 p0 = noise_pair(salted_seed, leaf.key, pair_base, mode);
        float2 p1 = noise_pair(salted_seed, leaf.key, pair_base + 1ul, mode);
        float2 p2 = noise_pair(salted_seed, leaf.key, pair_base + 2ul, mode);
        float2 p3 = noise_pair(salted_seed, leaf.key, pair_base + 3ul, mode);
        float mu8 = 0.125f * ((p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y));
        float2 raw = sub == 0u ? p0 : (sub == 1u ? p1 : (sub == 2u ? p2 : p3));
        float2 unit_dir = 1.05220856f * (raw - mu8) + 0.50f * float2(mu8);
        return step * unit_dir;
    }
    bool center_fiber = family_code == 0u
        || family_code == 5u
        || family_code == 7u
        || family_code == 9u
        || family_code >= 10u;
    if (center_fiber && leaf_pairs >= 2ul) {
        ulong pair_base = min(pair & ~1ul, leaf_pairs - 2ul);
        uint sub = uint(pair - pair_base);
        float2 q0 = noise_pair(salted_seed, leaf.key, pair_base, mode);
        float2 q1 = noise_pair(salted_seed, leaf.key, pair_base + 1ul, mode);
        float mu4 = 0.25f * ((q0.x + q0.y) + (q1.x + q1.y));
        float2 raw = sub == 0u ? q0 : q1;
        float2 unit_dir = 1.15470054f * (raw - mu4);
        return step * unit_dir;
    }
    float2 q = noise_pair(salted_seed, leaf.key, pair, mode);
    return step * q;
}

inline float2 proposal_pair_continuous(float2 base_value, Leaf leaf, ulong seed,
    ulong basis_seed, uint candidate, uint program, float radius, uint mode,
    ulong pair) {
    if (program == 4u)
        return sym_pair_continuous(base_value, leaf, seed ^ 0x8ebc6af09c88c6e3ul, radius, mode, pair);

    float2 noise = noise_pair(seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, mode);
    if (program == 3u) {
        float r_sq = base_value.x * base_value.x + base_value.y * base_value.y;
        float inv_r = r_sq > 1e-12f ? rsqrt(r_sq) : 0.0f;
        float2 tangent = float2(base_value.y * inv_r, -base_value.x * inv_r);
        uint slot = candidate & 3u;
        float alpha = slot == 0u ? 0.0f : (slot == 1u ? 0.5235987756f : (slot == 2u ? 0.7853981634f : 1.0471975512f));
        noise = cos(alpha) * noise + sin(alpha) * tangent;
        return (leaf.scale * radius) * noise;
    }
    if (program == 5u) {
        float s = leaf.scale * radius;
        float kappa = 0.5f * (base_value.x * noise.y - base_value.y * noise.x);
        float theta = (s * s) * kappa;
        float c = cos(theta);
        float s_sin = sin(theta);
        float x_rot = c * base_value.x - s_sin * base_value.y;
        float y_rot = s_sin * base_value.x + c * base_value.y;
        return float2(x_rot - base_value.x, y_rot - base_value.y);
    }
    return (leaf.scale * radius) * noise;
}

inline ushort2 sym_pair(
    float2 base_value,
    Leaf leaf,
    ulong salted_seed,
    float radius,
    uint mode,
    ulong pair) {
    float step = leaf.scale * radius;
    if (step == 0.0f) return encode_pair(base_value);
    uint family_code = leaf.address & 31u;
    ulong leaf_pairs = (leaf.length + 1ul) / 2ul;
    if (family_code == 4u && (pair >> 17) < 64ul && leaf_pairs >= 4ul) {
        ulong pair_base = min(pair & ~3ul, leaf_pairs - 4ul);
        uint sub = uint(pair - pair_base);
        float2 p0 = noise_pair(salted_seed, leaf.key, pair_base, mode);
        float2 p1 = noise_pair(salted_seed, leaf.key, pair_base + 1ul, mode);
        float2 p2 = noise_pair(salted_seed, leaf.key, pair_base + 2ul, mode);
        float2 p3 = noise_pair(salted_seed, leaf.key, pair_base + 3ul, mode);
        float mu8 = 0.125f * ((p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y));
        float2 raw = sub == 0u ? p0 : (sub == 1u ? p1 : (sub == 2u ? p2 : p3));
        float2 unit_dir = 1.05220856f * (raw - mu8) + 0.50f * float2(mu8);
        return encode_pair(base_value + step * unit_dir);
    }
    bool center_fiber = family_code == 0u
        || family_code == 5u
        || family_code == 7u
        || family_code == 9u
        || family_code >= 10u;
    if (center_fiber && leaf_pairs >= 2ul) {
        ulong pair_base = min(pair & ~1ul, leaf_pairs - 2ul);
        uint sub = uint(pair - pair_base);
        float2 q0 = noise_pair(salted_seed, leaf.key, pair_base, mode);
        float2 q1 = noise_pair(salted_seed, leaf.key, pair_base + 1ul, mode);
        float mu4 = 0.25f * ((q0.x + q0.y) + (q1.x + q1.y));
        float2 raw = sub == 0u ? q0 : q1;
        float2 unit_dir = 1.15470054f * (raw - mu4);
        return encode_pair(base_value + step * unit_dir);
    }
    float2 q = noise_pair(salted_seed, leaf.key, pair, mode);
    return encode_pair(base_value + step * q);
}

// Seed is the canonical candidate stream, before the single noise-domain salt.
// Every FP16 materialization path and sampled observation uses this operation.
inline ushort2 proposal_pair(float2 base_value, Leaf leaf, ulong seed,
    ulong basis_seed, uint candidate, uint program, float radius, uint mode,
    ulong pair) {
    if (program == 4u)
        return sym_pair(base_value, leaf, seed ^ 0x8ebc6af09c88c6e3ul, radius, mode, pair);

    float2 noise = noise_pair(seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, mode);
    if (program == 3u) {
        float r_sq = base_value.x * base_value.x + base_value.y * base_value.y;
        float inv_r = r_sq > 1e-12f ? rsqrt(r_sq) : 0.0f;
        float2 tangent = float2(base_value.y * inv_r, -base_value.x * inv_r);
        uint slot = candidate & 3u;
        float alpha = slot == 0u ? 0.0f : (slot == 1u ? 0.5235987756f : (slot == 2u ? 0.7853981634f : 1.0471975512f));
        noise = cos(alpha) * noise + sin(alpha) * tangent;
        return encode_pair(base_value + (leaf.scale * radius) * noise);
    }
    if (program == 5u) {
        // Exact SO(2) Givens pair rotation: theta = (s * s) * kappa where kappa = 0.5 * (u_i * v_j - v_i * u_j)
        float s = leaf.scale * radius;
        float kappa = 0.5f * (base_value.x * noise.y - base_value.y * noise.x);
        float theta = (s * s) * kappa;
        float c = cos(theta);
        float s_sin = sin(theta);
        float x_rot = c * base_value.x - s_sin * base_value.y;
        float y_rot = s_sin * base_value.x + c * base_value.y;
        return encode_pair(float2(x_rot, y_rot));
    }
    return encode_pair(base_value + (leaf.scale * radius) * noise);
}

#ifdef PROCEDURAL_SLOTS
inline ushort2 procedural_pair(float2 base_value, Leaf leaf, constant Params &p,
    ulong pair,
    uint candidate) {
    return proposal_pair(base_value, leaf,
        proposal_seed(p.stream_seed, candidate, p.program), p.basis_seed,
        candidate, p.program,
        proposal_radius(candidate, p.program, p.radius, p.alternate_radius),
        p.mode, pair);
}
#endif
#endif

inline ushort proposed_value(
    device const ushort *base,
    ulong index,
    Leaf leaf,
    device const ushort *reference,
    float inverse_rms,
    float noise,
    uint candidate,
    float radius) {
    float d = direction(decode(reference[index]), inverse_rms, noise, candidate);
    return encode(decode(base[index]) + (leaf.scale * radius) * d);
}
inline ushort proposed_value_bits(
    ushort base_bits,
    ushort reference_bits,
    Leaf leaf,
    float inverse_rms,
    float noise,
    uint candidate,
    float radius) {
    float d = direction(decode(reference_bits), inverse_rms, noise, candidate);
    return encode(decode(base_bits) + (leaf.scale * radius) * d);
}

kernel void bf16_propose(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const ushort *reference [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device const Leaf *leaves [[buffer(5)]],
    device const Tile *tiles [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    device float *geometry [[buffer(9)]],
    constant Params &p [[buffer(10)]],
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
        ulong seed = p.program == 3u
            ? proposal_seed(p.stream_seed, p.candidate, p.program) : p.seed;
        ushort2 values = proposal_pair(decode_pair(base_pair), leaf, seed,
            p.basis_seed, p.candidate, p.program, p.radius, p.mode,
            element / 2ul);
        float base_value = decode(base[index]);
        ushort value = values[element & 1ul];
#else
        float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
        float base_value = decode(base[index]);
        float d = direction(decode(reference[index]), scales[tile.leaf], noise, p.candidate);
        ushort value = encode(base_value + (leaf.scale * p.radius) * d);
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
kernel void bf16_propose_initial(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const ushort *reference [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device const Leaf *leaves [[buffer(5)]],
    device const Tile *tiles [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    device float *geometry [[buffer(9)]],
    constant Params &p [[buffer(10)]],
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
        ulong seed = p.program == 3u
            ? proposal_seed(p.stream_seed, p.candidate, p.program) : p.seed;
        ushort2 values = proposal_pair(base_values, leaf, seed, p.basis_seed,
            p.candidate, p.program, p.radius, p.mode, element / 2ul);
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
        float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
        ushort value = encode(base_value + (leaf.scale * p.radius) * noise);
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


kernel void bf16_propose_pool(
    device const ushort *base [[buffer(0)]],
    device const ushort *anchor [[buffer(1)]],
    device const ushort *rejected [[buffer(2)]],
    device const ushort *reference [[buffer(3)]],
    device const float *scales [[buffer(4)]],
    device const Leaf *leaves [[buffer(5)]],
    device const Tile *tiles [[buffer(6)]],
    device ushort *output [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    device float *geometry [[buffer(9)]],
    constant Params &p [[buffer(10)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
#ifndef FP16_INDEPENDENT
    float inverse_rms = scales[tile.leaf];
#endif
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    float dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float reference_norm = 0.0f;
    float reference_dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        float2 persistent_pair = noise_pair(persistent_seed, leaf.key, pair, p.mode);
        float2 fresh_pair = noise_pair(fresh_seed, leaf.key, pair, p.mode);
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
        if (p.program == 0u) {
            float radius = leaf.scale * p.radius;
            float alternate_radius = leaf.scale * p.alternate_radius;
            values[0] = encode_pair(base_values + radius * persistent_pair);
            values[1] = encode_pair(base_values + alternate_radius * persistent_pair);
            values[2] = encode_pair(base_values + radius * fresh_pair);
            values[3] = encode_pair(base_values + alternate_radius * fresh_pair);
        } else {
#ifdef PROCEDURAL_SLOTS
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++)
                values[candidate] = procedural_pair(
                    base_values,
                    leaf,
                    p,
                    pair,
                    candidate);
#else
            for (uint candidate = 0; candidate < 4; candidate++)
                values[candidate] = proposal_pair(
                    base_values,
                    leaf,
                    proposal_seed(p.stream_seed, candidate, p.program),
                    p.basis_seed,
                    candidate,
                    p.program,
                    candidate % 2 == 0 ? p.radius : p.alternate_radius,
                    p.mode,
                    pair);
#endif
        }
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
            float persistent = pair_offset == 0 ? persistent_pair.x : persistent_pair.y;
            float fresh = pair_offset == 0 ? fresh_pair.x : fresh_pair.y;
            ushort base_bits = base[index];
#ifdef FP16_INDEPENDENT
            ushort reference_bits = 0;
#else
            ushort reference_bits = reference[index];
#endif
            float base_value = decode(base_bits);
            float anchor_value = base_value;
            if (p.base_slot != 0) anchor_value = decode(anchor[index]);
            float reference_value = decode(reference_bits);
            float rejected_value = 0.0f;
            if (p.history == 2 && p.base_slot != 1)
                rejected_value = decode(rejected[index]);
            else if (p.history == 2)
                rejected_value = base_value;
            float persistent_direction = direction(reference_value, inverse_rms, persistent, 0);
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            float reference_delta = leaf.scale * reference_value * inverse_rms;
            reference_norm += reference_delta * reference_delta * leaf.weight;
#endif
            float radius = leaf.scale * p.radius;
            float alternate_radius = leaf.scale * p.alternate_radius;
            ushort values[4] = {
                encode(base_value + radius * persistent_direction),
                encode(base_value + alternate_radius * persistent_direction),
                encode(base_value + radius * fresh),
                encode(base_value + alternate_radius * fresh),
            };
#ifdef PROCEDURAL_SLOTS
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                float2 samples = noise_pair(candidate_seed(p.stream_seed, candidate)
                    ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, p.mode);
                float noise = samples[pair_offset];
                float candidate_radius = proposal_radius(
                    candidate, p.program, p.radius, p.alternate_radius);
                values[candidate] = proposed_value_bits(base_bits, reference_bits, leaf,
                    inverse_rms, noise, candidate, candidate_radius);
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
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                reference_dots[candidate] += delta * reference_delta * leaf.weight;
#endif
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
    threadgroup float reduced_a[32], reduced_b[32], reduced_sq[32];
    threadgroup uint reduced_counts[32], reduced_flags[32];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        float a_sum = simd_sum(a[candidate]);
        float b_sum = simd_sum(b[candidate]);
        float sq_sum = simd_sum(sq[candidate]);
        uint changed_sum = simd_sum(changed[candidate]);
        uint invalid_sum = simd_sum(invalid[candidate]);
        if (lane == 0) {
            uint slot = candidate * 8 + simdgroup;
            reduced_a[slot] = a_sum;
            reduced_b[slot] = b_sum;
            reduced_sq[slot] = sq_sum;
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
            uint changed_sum = simd_sum(active ? reduced_counts[slot] : 0u);
            uint invalid_sum = simd_sum(active ? reduced_flags[slot] : 0u);
            if (lane != 0) continue;
            partials[(p.initialize * 4 + candidate) * p.tiles + tile_index] =
                {a_sum, b_sum, sq_sum, changed_sum, uint(invalid_sum != 0)};
#ifndef POOL_GEOMETRY_DIAGNOSTICS
            if (p.base_slot == 0)
                geometry[candidate * p.tiles + tile_index] = a_sum;
            else if (p.base_slot == 1)
                geometry[candidate * p.tiles + tile_index] = b_sum;
#endif
        }
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float sums_a[1024], sums_b[1024], sums_sq[1024], sums_ref[768];
    for (uint candidate = 0; candidate < 4; candidate++)
        sums_a[candidate * 256 + tid] = norms[candidate];
    for (uint pair = 0; pair < 4; pair++)
        sums_b[pair * 256 + tid] = dots[pair];
    for (uint pair = 0; pair < 2; pair++)
        sums_sq[pair * 256 + tid] = dots[pair + 4];
    sums_sq[2 * 256 + tid] = reference_norm;
    sums_sq[3 * 256 + tid] = reference_dots[0];
    for (uint candidate = 1; candidate < 4; candidate++)
        sums_ref[(candidate - 1) * 256 + tid] = reference_dots[candidate];
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
            for (uint pair = 0; pair < 4; pair++) {
                uint slot = pair * 256 + tid;
                sums_sq[slot] += sums_sq[slot + stride];
            }
            for (uint candidate = 0; candidate < 3; candidate++) {
                uint slot = candidate * 256 + tid;
                sums_ref[slot] += sums_ref[slot + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        for (uint candidate = 0; candidate < 4; candidate++)
            geometry[candidate * p.tiles + tile_index] = sums_a[candidate * 256];
        for (uint pair = 0; pair < 4; pair++)
            geometry[(pair + 4) * p.tiles + tile_index] = sums_b[pair * 256];
        for (uint pair = 0; pair < 2; pair++)
            geometry[(pair + 8) * p.tiles + tile_index] = sums_sq[pair * 256];
        geometry[10 * p.tiles + tile_index] = sums_sq[2 * 256];
        geometry[11 * p.tiles + tile_index] = sums_sq[3 * 256];
        for (uint candidate = 1; candidate < 4; candidate++)
            geometry[(candidate + 11) * p.tiles + tile_index]
                = sums_ref[(candidate - 1) * 256];
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
    if (q.distance_scaling != 0) {
        float query_neighbors[BAUX_HISTORY];
        uint query_count = 0;
        for (uint i = 0; i < q.history; i++) {
            if (i == excluded_index) continue;
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
            float s_q = sqrt(query_scale), s_h = sqrt(max(q.local_scales[i], 1.0e-12f)), hyp = float(q.distance_scaling >> 1);
            scaled_distances[i] = (distances[i] + hyp * (s_q - s_h) * (s_q - s_h)) / ((1.0f + hyp) * s_q * s_h);
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
            float risk = q.epistemic_scale * scaled_distances[index] + sqrt(variance_error);
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
kernel void bf16_posterior_probe(
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

// For the short-history regime, spend every lane on distinct coordinates and
// reconstruct all historical observations in registers.  This avoids leaving
// most lanes idle when the replay lineage contains only a handful of rows.
kernel void bf16_replay_short(
    device ushort *lineage [[buffer(0)]],
    device const ushort *base [[buffer(1)]],
    device const Leaf *leaves [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device const ReplayStep *steps [[buffer(4)]],
    device const float *historical_scales [[buffer(5)]],
    device const float *base_weights [[buffer(6)]],
    device float *partials [[buffer(7)]],
    device float *components [[buffer(8)]],
    device Partial *pool_partials [[buffer(9)]],
    device float *pool_geometry [[buffer(10)]],
    constant Params &p [[buffer(11)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
#ifndef FP16_INDEPENDENT
    return;
#else
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    uint active_rows = min(16u, p.history - p.initialize);
    float sums[64] = {0.0f};
    float component_sums[64] = {0.0f};
    float proposal_norms[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    float proposal_dots[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#endif
    float proposal_squared[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint proposal_changed[4] = {0u, 0u, 0u, 0u};
    uint proposal_invalid[4] = {0u, 0u, 0u, 0u};
    ulong persistent_seed = p.seed ^ 0x8ebc6af09c88c6e3ul;
    ulong fresh_seed = p.stream_seed ^ 0x8ebc6af09c88c6e3ul;
    ulong2 persistent_domain = ziggurat_domain(persistent_seed, leaf.key);
    ulong2 fresh_domain = ziggurat_domain(fresh_seed, leaf.key);
    ulong2 history_domains[16] = {ulong2(0ul)};
    for (uint local = 0u; local < active_rows; local++) {
        uint row = p.initialize + local;
        history_domains[local] = ziggurat_domain(
            steps[row].seed ^ 0x8ebc6af09c88c6e3ul,
            leaf.key);
    }
    uint pair_count = (tile.length + 1u) / 2u;
    for (uint pair = tid; pair < pair_count; pair += 256u) {
        uint item = pair * 2u;
        bool has_second = item + 1u < tile.length;
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        ushort2 current_lineage = ushort2(
            lineage[index],
            has_second ? lineage[index + 1ul] : 0u);
        ushort2 base_bits = ushort2(base[index], has_second ? base[index + 1ul] : 0u);
        float2 base_value = decode_pair(base_bits);
        ushort2 candidates[4];
        if (p.program == 0u) {
#ifdef RADEMACHER_ONLY
            float2 persistent = rademacher_pair(persistent_domain, element / 2ul);
            float2 fresh = rademacher_pair(fresh_domain, element / 2ul);
#else
            float2 persistent = p.mode == 1u
                ? rademacher_pair(persistent_domain, element / 2ul)
                : ziggurat_pair_domain(persistent_domain, element / 2ul);
            float2 fresh = p.mode == 1u
                ? rademacher_pair(fresh_domain, element / 2ul)
                : ziggurat_pair_domain(fresh_domain, element / 2ul);
#endif
            candidates[0] = encode_pair(base_value + leaf.scale * p.radius * persistent);
            candidates[1] = encode_pair(base_value + leaf.scale * p.alternate_radius * persistent);
            candidates[2] = encode_pair(base_value + leaf.scale * p.radius * fresh);
            candidates[3] = encode_pair(base_value + leaf.scale * p.alternate_radius * fresh);
        } else {
#ifdef PROCEDURAL_SLOTS
            for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
                float radius = proposal_radius(
                    candidate, p.program, p.radius, p.alternate_radius);
                candidates[candidate] = proposal_pair(
                    base_value, leaf,
                    proposal_seed(p.stream_seed, candidate, p.program), p.basis_seed,
                    candidate, p.program, radius, p.mode,
                    element / 2ul);
            }
#else
            for (uint candidate = 0; candidate < 4; candidate++) {
                float radius = proposal_radius(
                    candidate, p.program, p.radius, p.alternate_radius);
                candidates[candidate] = proposal_pair(
                    base_value, leaf,
                    proposal_seed(p.stream_seed, candidate, p.program), p.basis_seed,
                    candidate, p.program, radius, p.mode,
                    element / 2ul);
            }
#endif
        }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
        float2 candidate_deltas[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
#endif
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
            float2 delta = decode_pair(candidates[candidate]) - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
            candidate_deltas[candidate] = delta;
#endif
            float2 squared = delta * delta;
            float pair_squared = squared.x + (has_second ? squared.y : 0.0f);
            proposal_norms[candidate] += pair_squared * leaf.weight;
            proposal_squared[candidate] += pair_squared;
            proposal_changed[candidate] += uint(candidates[candidate].x != base_bits.x)
                + uint(has_second && candidates[candidate].y != base_bits.y);
            proposal_invalid[candidate] |= uint(invalid_value(candidates[candidate].x))
                | uint(has_second && invalid_value(candidates[candidate].y));
        }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
        const uint2 candidate_pairs[6] = {
            uint2(0u, 1u), uint2(0u, 2u), uint2(0u, 3u),
            uint2(1u, 2u), uint2(1u, 3u), uint2(2u, 3u),
        };
        for (uint dot = 0; dot < 6u; dot++) {
            float2 product = candidate_deltas[candidate_pairs[dot].x]
                * candidate_deltas[candidate_pairs[dot].y];
            proposal_dots[dot] += (product.x + (has_second ? product.y : 0.0f))
                * leaf.weight;
        }
#endif
        for (uint local = 0; local < active_rows; local++) {
            uint row = p.initialize + local;
            ushort2 historical = current_lineage;
            if (row != 0u) {
                ReplayStep step = steps[row];
                if (p.program == 0u) {
#ifdef RADEMACHER_ONLY
                    float2 noise = rademacher_pair(
                        history_domains[local], element / 2ul);
#else
                    float2 noise = step.mode == 1u
                        ? rademacher_pair(history_domains[local], element / 2ul)
                        : ziggurat_pair_domain(history_domains[local], element / 2ul);
#endif
                    float scale = historical_scales[row * p.blocks + tile.leaf];
                    historical = encode_pair(
                        decode_pair(current_lineage) + scale * step.radius * noise);
                } else {
                    float scale = historical_scales[row * p.blocks + tile.leaf];
                    Leaf hist_leaf = leaf;
                    hist_leaf.scale = scale;
                    historical = proposal_pair(
                        decode_pair(current_lineage), hist_leaf, step.seed, p.basis_seed,
                        step.candidate, p.program, step.radius, step.mode,
                        element / 2ul);
                }
            }
            float2 historical_value = decode_pair(historical);
            for (uint candidate = 0; candidate < 4; candidate++) {
                float2 delta = decode_pair(candidates[candidate]) - historical_value;
                float2 squared = delta * delta;
                float pair_squared = squared.x + (has_second ? squared.y : 0.0f);
                uint metric = local * 4u + candidate;
                sums[metric] += pair_squared * leaf.weight;
                component_sums[metric] += pair_squared * base_weights[tile.leaf];
            }
            if (steps[row].accepted != 0u) current_lineage = historical;
        }
        lineage[index] = current_lineage.x;
        if (has_second) lineage[index + 1ul] = current_lineage.y;
    }
    threadgroup float weighted_groups[8 * 64];
    threadgroup float component_groups[8 * 64];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    uint metrics = active_rows * 4u;
    for (uint metric = 0; metric < metrics; metric++) {
        float weighted = simd_sum(sums[metric]);
        float component = simd_sum(component_sums[metric]);
        if (lane == 0u) {
            weighted_groups[simdgroup * 64u + metric] = weighted;
            component_groups[simdgroup * 64u + metric] = component;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < metrics) {
        float weighted = 0.0f;
        float component = 0.0f;
        for (uint simd = 0; simd < 8u; simd++) {
            weighted += weighted_groups[simd * 64u + tid];
            component += component_groups[simd * 64u + tid];
        }
        uint local = tid / 4u;
        uint row = p.initialize + local;
        uint candidate = tid - local * 4u;
        ulong output = (ulong(candidate) * ulong(BAUX_HISTORY) + row) * p.tiles + tile_index;
        partials[output] = weighted;
        components[output] = component;
    }
    threadgroup float norm_groups[8 * 4];
    threadgroup float squared_groups[8 * 4];
    threadgroup uint changed_groups[8 * 4];
    threadgroup uint invalid_groups[8 * 4];
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        float norm = simd_sum(proposal_norms[candidate]);
        float squared = simd_sum(proposal_squared[candidate]);
        uint changed = simd_sum(proposal_changed[candidate]);
        uint invalid = simd_sum(proposal_invalid[candidate]);
        if (lane == 0u) {
            uint slot = simdgroup * 4u + candidate;
            norm_groups[slot] = norm;
            squared_groups[slot] = squared;
            changed_groups[slot] = changed;
            invalid_groups[slot] = invalid;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < PROCEDURAL_COUNT) {
        float norm = 0.0f;
        float squared = 0.0f;
        uint changed = 0u;
        uint invalid = 0u;
        for (uint simd = 0; simd < 8u; simd++) {
            uint slot = simd * 4u + tid;
            norm += norm_groups[slot];
            squared += squared_groups[slot];
            changed += changed_groups[slot];
            invalid |= invalid_groups[slot];
        }
        pool_partials[tid * p.tiles + tile_index] = {
            0.0f, 0.0f, squared, changed, invalid
        };
        pool_geometry[tid * p.tiles + tile_index] = norm;
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float dot_groups[8 * 6];
    for (uint dot = 0; dot < 6u; dot++) {
        float value = simd_sum(proposal_dots[dot]);
        if (lane == 0u) dot_groups[simdgroup * 6u + dot] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6u) {
        float value = 0.0f;
        for (uint simd = 0; simd < 8u; simd++)
            value += dot_groups[simd * 6u + tid];
        pool_geometry[(tid + 4u) * p.tiles + tile_index] = value;
    }
    if (tid < 5u)
        pool_geometry[(tid + 10u) * p.tiles + tile_index] = 0.0f;
#endif
#endif
}

kernel void bf16_reduce_replay(
    device const float *partials [[buffer(0)]],
    device const float *components [[buffer(1)]],
    device const uint *family_groups [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device float *distances [[buffer(4)]],
    device float *family_distances [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    uint output [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint candidate = output / p.history;
    uint row = output - candidate * p.history;
    float total = 0.0f;
    float families[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ulong source = (ulong(candidate) * ulong(BAUX_HISTORY) + row) * p.tiles;
    for (uint tile = tid; tile < p.tiles; tile += 256u) {
        total += partials[source + tile];
        uint family = min(family_groups[tiles[tile].leaf], 3u);
        families[family] += components[source + tile];
    }
    threadgroup float reduced[5 * 256];
    reduced[tid] = total;
    for (uint family = 0; family < 4; family++)
        reduced[(family + 1u) * 256u + tid] = families[family];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride)
            for (uint value = 0; value < 5; value++)
                reduced[value * 256u + tid] += reduced[value * 256u + tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        distances[candidate * p.history + row] = reduced[0];
        ulong family_output = (ulong(candidate) * p.history + row) * 4ul;
        for (uint family = 0; family < 4; family++)
            family_distances[family_output + family] = reduced[(family + 1u) * 256u];
    }
}

// Collapse the per-tile records in parallel before the deliberately scalar
// acquisition step.  The old selector walked every tile on one GPU thread.
kernel void bf16_reduce_pool(
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

struct NeighborPoint { ulong element; uint leaf; float mass; };

kernel void bf16_sample(
    device const ushort *base [[buffer(0)]],
    device const Leaf *leaves [[buffer(1)]],
    device const NeighborPoint *points [[buffer(2)]],
    device ushort *pool [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    device float *pool_deltas [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.tiles) return;
    NeighborPoint point = points[index];
    Leaf leaf = leaves[point.leaf];
#ifdef FP16_INDEPENDENT
    ulong first = (point.element / 2ul) * 2ul;
    ushort2 base_pair = ushort2(base[leaf.offset + first],
        first + 1ul < leaf.length ? base[leaf.offset + first + 1ul] : 0);
    float2 base_value = decode_pair(base_pair);
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        ushort2 values = proposal_pair(base_value, leaf,
            proposal_seed(p.stream_seed, candidate, p.program), p.basis_seed,
            candidate, p.program,
            proposal_radius(candidate, p.program, p.radius, p.alternate_radius),
            p.mode, point.element / 2ul);
        pool[candidate * p.tiles + index] = values[point.element & 1ul];
        float2 deltas = proposal_pair_continuous(base_value, leaf,
            proposal_seed(p.stream_seed, candidate, p.program), p.basis_seed,
            candidate, p.program,
            proposal_radius(candidate, p.program, p.radius, p.alternate_radius),
            p.mode, point.element / 2ul);
        pool_deltas[candidate * p.tiles + index] = deltas[point.element & 1ul];
    }
#else
    float value = decode(base[leaf.offset + point.element]);
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        float noise = sample(candidate_seed(p.seed, candidate)
            ^ 0x8ebc6af09c88c6e3ul, leaf.key, point.element, p.mode);
        float radius = proposal_radius(
            candidate, p.program, p.radius, p.alternate_radius);
        float delta = (leaf.scale * radius) * noise;
        pool[candidate * p.tiles + index] = encode(value + delta);
        pool_deltas[candidate * p.tiles + index] = delta;
    }
#endif
}

kernel void bf16_sample_distances(
    device const ushort *pool [[buffer(0)]],
    device const ushort *history [[buffer(1)]],
    device const NeighborPoint *points [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device float *distances [[buffer(4)]],
    device const float *base_weights [[buffer(5)]],
    device const uint *groups [[buffer(6)]],
    device float *components [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint candidate = pair / p.history;
    uint row = pair % p.history;
    float sum = 0.0f;
    float grouped[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint index = tid; index < p.tiles; index += 256) {
        NeighborPoint point = points[index];
        float delta = decode(pool[candidate * p.tiles + index]) - decode(history[row * p.tiles + index]);
        sum += delta * delta * (point.mass * leaves[point.leaf].weight);
        if (p.initialize != 0)
            grouped[groups[point.leaf]] += delta * delta * (point.mass * base_weights[point.leaf]);
    }
    threadgroup float partial[256];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) distances[pair] = partial[0];
    if (p.initialize == 0) return;
    for (uint group = 0; group < 4; group++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        partial[tid] = grouped[group];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride != 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) components[pair * 4 + group] = partial[0];
    }
}

kernel void bf16_select(
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
        proposal_norms[candidate] = q.latent_history != 0
            ? q.latent_norms[candidate]
            : aggregates[candidate].squared;
        for (uint row = 0; row < q.history; row++) {
            distances[candidate][row] = q.exact_history != 0 || q.sampled_history != 0
                ? pool_distances[candidate * q.history + row]
                : q.implicit_history != 0
                ? q.base_distances[row] + proposal_norms[candidate]
                : 0.0f;
            if (q.axis_enabled != 0) {
                float delta = q.axis_candidates[candidate] - q.axis_history[row];
                distances[candidate][row] += q.axis_weight * delta * delta;
            }
        }
        if (q.exact_history != 0 || q.sampled_history != 0) {
            proposal_norms[candidate] = distances[candidate][q.incumbent_index];
            changed[candidate] = proposal_norms[candidate] > 0.0f;
            for (uint row = 0; row < q.history; row++)
                invalid[candidate] = invalid[candidate]
                    || !isfinite(distances[candidate][row]);
        }
    }
    if (q.latent_history != 0 && q.sampled_history == 0) {
        for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
            changed[candidate] = proposal_norms[candidate] > 0.0f;
            invalid[candidate] = !isfinite(proposal_norms[candidate]);
        }
    }
    if (q.exact_history == 0 && q.latent_history == 0 && q.sampled_history == 0)
    for (uint candidate = 0; candidate < PROCEDURAL_COUNT; candidate++) {
        for (uint row = 0; row < q.resident_history; row += 2) {
            Partial aggregate = aggregates[(row / 2) * 4 + candidate];
            float first = aggregate.anchor;
            float second = aggregate.rejected;
            changed[candidate] = changed[candidate] || aggregate.changed != 0;
            invalid[candidate] = invalid[candidate] || aggregate.invalid != 0;
            uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
            if (q.exact_history == 0 && q.latent_history == 0)
                distances[candidate][first_index] = first;
            if (row + 1 < q.resident_history) {
                uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
                if (q.exact_history == 0 && q.latent_history == 0)
                    distances[candidate][second_index] = second;
            }
            invalid[candidate] = invalid[candidate] || !isfinite(first)
                || (row + 1 < q.resident_history && !isfinite(second));
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
        q.incumbent_index,
        incumbent_mean,
        incumbent_standard_error,
        incumbent_neighbors);
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
        if (q.forced_candidate < 4) {
            if (candidate == q.forced_candidate) {
                selected = candidate;
                best = 0.0f;
            }
            continue;
        }
        if (isfinite(score) && (selected == 4 || score > best)) {
            selected = candidate;
            best = score;
        }
    }
    decision->index = selected;
    decision->valid = selected < 4 ? 1u : 0u;
    decision->root_seed = q.root_seed;
    decision->seed = selected < 4 ? candidate_seed(q.root_seed, selected) : 0ul;
    decision->basis_seed = q.basis_seed;
    decision->radius = selected < 4
        ? proposal_radius(selected, q.program, q.radius, q.alternate_radius)
        : 0.0f;
    decision->score = selected < 4 ? best : 0.0f;
    decision->mode = q.mode;
    decision->program = q.program;
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

kernel void bf16_materialize(
    device const ushort *base [[buffer(0)]],
    device const ushort *reference [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device const Leaf *leaves [[buffer(3)]],
    device const Tile *tiles [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    device const Decision *decision [[buffer(6)]],
    constant uint &account [[buffer(7)]],
    device Partial *partials [[buffer(8)]],
    constant Params &p [[buffer(9)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (decision->valid == 0) return;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
#ifdef FP16_INDEPENDENT
    ulong seed = decision->program == 3u
        ? proposal_seed(decision->root_seed, decision->index, decision->program)
        : decision->seed;
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
        ushort2 result = proposal_pair(decode_pair(base_pair), leaf, seed,
            decision->basis_seed, decision->index, decision->program,
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
        float2 samples = noise_pair(seed, leaf.key, pair, decision->mode);
        float inverse_rms = scales[tile.leaf];

        ushort first_bits = base[index];
        float first_direction = decision->index < 2
            ? direction(decode(reference[index]), inverse_rms, samples.x, decision->index)
            : samples.x;
        output[index] = encode(decode(first_bits) + step * first_direction);
        if (account != 0u) {
            ushort value = output[index];
            changed += uint(value != first_bits);
            invalid |= uint(invalid_value(value));
            float delta = decode(value) - decode(first_bits);
            squared += delta * delta;
        }

        if (item + 1 < tile.length) {
            ushort second_bits = base[index + 1];
            float second_direction = decision->index < 2
                ? direction(
                    decode(reference[index + 1]),
                    inverse_rms,
                    samples.y,
                    decision->index)
                : samples.y;
            output[index + 1] = encode(decode(second_bits) + step * second_direction);
            if (account != 0u) {
                ushort value = output[index + 1];
                changed += uint(value != second_bits);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - decode(second_bits);
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

kernel void bf16_reference(
    device ushort *reference [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const Leaf *leaves [[buffer(2)]],
    device const Tile *tiles [[buffer(3)]],
    device float *partials [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sum = 0.0f;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        float value;
        if (p.initialize != 0) {
            value = sample(p.seed ^ 0xe7037ed1a0b428dbul, leaf.key, element, p.mode);
        } else {
            float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
            value = direction(decode(reference[index]), scales[tile.leaf], noise, p.candidate);
        }
        ushort bits = encode(value);
        reference[index] = bits;
        float stored = decode(bits);
        sum += stored * stored;
    }
    threadgroup float sums[256];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[tile_index] = sums[0];
}

kernel void bf16_reference_rms(
    device const float *partials [[buffer(0)]],
    device const Leaf *leaves [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device float *scales [[buffer(3)]],
    uint leaf [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint tile = offsets[leaf] + tid; tile < offsets[leaf + 1]; tile += 256)
        sum += partials[tile];
    threadgroup float sums[256];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scales[leaf] = sqrt(float(leaves[leaf].length) / sums[0]);
}
