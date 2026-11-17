#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

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
struct DimMeta {
    float weight;
    float tr_half_scale;
    uint tied_leader;
    uint categories;
};
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
    uint pool_count;
    uint pool_slots;
    uint dimensions;
    uint num_pert;
    uint sobol_offset;
    uint lhd_points;
    uint box_bounded;
    uint has_dim_meta;
    uint coord_system;
    uint pad;
};
struct SelectionParams {
    ulong root_seed;
    float outcomes[BAUX_HISTORY];
    float variances[BAUX_HISTORY];
    float draws[BAUX_HISTORY];
    float base_distances[BAUX_HISTORY];
    float local_scales[BAUX_HISTORY];
    float axis_history[BAUX_HISTORY];
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
    uint pool_count;
    uint pool_slots;
    uint num_arms;
    uint noise_aware;
    uint y_bounds_mode;
    float y_lower;
    float y_upper;
    float affine_a;
    float affine_b;
    float affine_c;
    uint pad;
};
struct CandidateScore {
    float mean;
    float standard_error;
    float score;
    uint neighbors;
    uint changed;
    uint invalid;
};
struct Decision {
    uint index;
    uint valid;
    ulong root_seed;
    ulong seed;
    float radius;
    float score;
    uint mode;
    uint num_selected;
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
inline float decode(ushort x);

inline ulong mix64(ulong x) {
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    return x ^ (x >> 31);
}

kernel void import_partial(
    device const ushort *incoming [[buffer(0)]],
    device const ushort *first [[buffer(1)]],
    device const ushort *second [[buffer(2)]],
    device const ushort *incumbent [[buffer(3)]],
    device const Leaf *leaves [[buffer(4)]],
    device const Tile *tiles [[buffer(5)]],
    device Partial *partials [[buffer(6)]],
    constant Params &p [[buffer(7)]],
    device const DimMeta *dim_meta [[buffer(8)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float base_sum = 0.0f;
    for (uint item = tid; item < tile.length; item += 256) {
        ulong index = leaf.offset + ulong(tile.start) + ulong(item);
        float w = p.has_dim_meta != 0u ? dim_meta[index].weight : leaf.weight;
        float value = decode(incoming[index]);
        float delta = value - decode(first[index]);
        first_sum += w * delta * delta;
        if (p.history == 2) {
            delta = value - decode(second[index]);
            second_sum += w * delta * delta;
        }
        if (p.initialize == 0) {
            delta = value - decode(incumbent[index]);
            base_sum += w * delta * delta;
        }
    }
    threadgroup float first_parts[256];
    threadgroup float second_parts[256];
    threadgroup float base_parts[256];
    first_parts[tid] = first_sum;
    second_parts[tid] = second_sum;
    base_parts[tid] = base_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            first_parts[tid] += first_parts[tid + stride];
            second_parts[tid] += second_parts[tid + stride];
            base_parts[tid] += base_parts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[p.initialize * p.tiles + tile_index] = {
            first_parts[0], second_parts[0], base_parts[0], 0u, 0u
        };
    }
}

kernel void import_reduce(
    device const Partial *partials [[buffer(0)]],
    device float *distances [[buffer(1)]],
    constant uint &tiles [[buffer(2)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    float base_sum = 0.0f;
    for (uint tile = tid; tile < tiles; tile += 256) {
        Partial part = partials[pair * tiles + tile];
        first_sum += part.anchor;
        second_sum += part.rejected;
        base_sum += part.squared;
    }
    threadgroup float first_parts[256];
    threadgroup float second_parts[256];
    threadgroup float base_parts[256];
    first_parts[tid] = first_sum;
    second_parts[tid] = second_sum;
    base_parts[tid] = base_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            first_parts[tid] += first_parts[tid + stride];
            second_parts[tid] += second_parts[tid + stride];
            base_parts[tid] += base_parts[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        distances[pair * 2] = first_parts[0];
        distances[pair * 2 + 1] = second_parts[0];
        if (pair == 0) distances[2 * BAUX_HISTORY] = base_parts[0];
    }
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
inline ulong candidate_seed(ulong root, uint candidate, uint pool_slots) {
    uint arm = pool_slots > 0u ? (candidate / pool_slots) : 0u;
    uint slot = pool_slots > 0u ? (candidate % pool_slots) : candidate;
    if (arm != 0u) {
        ulong z = (root ^ 0x70726f6361726d31ul ^ ulong(arm)) + 0x9e3779b97f4a7c15ul;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
        root = z ^ (z >> 31);
    }
    uint stream = slot;
    uint low = uint(root);
    uint high = uint(root >> 32);
    return ulong(trial_hash(low, high, stream))
        | (ulong(trial_hash(high, low, stream ^ 0x9e3779b9u)) << 32);
}

// Stateless Key::fold, Owen scramble, and Feistel LHD helpers matching baux::semantics::pool.
struct PoolKey {
    ulong hi;
    ulong lo;
};

inline PoolKey pool_key(ulong seed) {
    ulong hi = mix64(seed);
    ulong lo = mix64(hi ^ 0xd2b74407b1ce6e93ul);
    return {hi, lo};
}

inline PoolKey key_fold(PoolKey k, ulong tag) {
    ulong hi = mix64(k.hi ^ (tag * 0x9e3779b97f4a7c15ul));
    ulong lo = mix64(k.lo ^ hi ^ (tag * 0xbf58476d1ce4e5b9ul));
    return {hi, lo};
}

inline ulong key_seed(PoolKey k) {
    return mix64(k.hi ^ ((k.lo << 31u) | (k.lo >> 33u)));
}

inline float key_uniform(PoolKey k) {
    return float(uint(key_seed(k) >> 40u)) * 0x1.0p-24f;
}

inline PoolKey cand_key(PoolKey k, uint cand) {
    return key_fold(key_fold(k, 0x43414e445f4b4559ul), ulong(cand));
}

inline uint forced_dim(PoolKey kc, uint dim) {
    return dim <= 1u ? 0u : uint(key_seed(key_fold(kc, 0x464f524345445f44ul)) % ulong(dim));
}

inline float hash_unit(PoolKey kc, uint j, ulong tag) {
    return key_uniform(key_fold(key_fold(kc, tag), ulong(j)));
}

inline float unit_draw(ulong seed, ulong key, ulong dim, uint sub) {
    ulong h = mix64(seed ^ mix64(key + dim * 0x9e3779b97f4a7c15ul + ulong(sub) * 0xd3a2646c9e3779b9ul));
    return min(float(uint(h >> 40)) * 0x1.0p-24f, 0.99999994f);
}

inline uint owen_scramble(uint v, uint seed) {
    uint x = reverse_bits(v);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return reverse_bits(x);
}

inline uint lhd_bin(PoolKey rk, uint sample, uint total, uint dim) {
    if (total <= 1u || sample >= total) return 0u;
    PoolKey kd = key_fold(key_fold(rk, 0x4c48445354524154ul), ulong(dim));
    uint bits = 32u - clz(total - 1u);
    uint a = bits / 2u;
    uint b = bits - a;
    uint mask_a = (1u << a) - 1u;
    uint mask_b = (1u << b) - 1u;
    uint x = sample;
    for (uint iter = 0u; iter < 128u; iter++) {
        uint u = x & mask_a;
        uint v = (x >> a) & mask_b;
        u ^= uint(mix64(kd.hi ^ ulong(v))) & mask_a;
        v ^= uint(mix64(kd.lo ^ ulong(u))) & mask_b;
        u ^= uint(mix64(((kd.hi << 17u) | (kd.hi >> 47u)) ^ ulong(v))) & mask_a;
        v ^= uint(mix64(((kd.lo << 29u) | (kd.lo >> 35u)) ^ ulong(u))) & mask_b;
        x = u | (v << a);
        if (x < total) return x;
    }
    return sample % total;
}

inline ushort sample_manifold(
    float base_value,
    Leaf leaf,
    ulong cand_seed,
    ulong stream_seed,
    uint candidate,
    float radius,
    uint mode,
    ulong global_index,
    constant Params &p,
    device const DimMeta *dim_meta,
    device const uint *sobol_dirs) {
    uint leader = uint(global_index);
    float half_scale = 1.0f;
    uint cats = 0u;
    if (p.has_dim_meta != 0u) {
        DimMeta dm = dim_meta[global_index];
        if (dm.tied_leader != 0xffffffffu) leader = dm.tied_leader;
        half_scale = dm.tr_half_scale;
        cats = dm.categories;
    }
    float proposed = base_value;
    if (mode <= 1u && p.coord_system == 0u) {
        float noise = sample(cand_seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, ulong(leader), mode);
        proposed = base_value + (leaf.scale * radius * half_scale) * noise;
        if (p.box_bounded != 0u) proposed = clamp(proposed, 0.0f, 1.0f);
    } else if (mode <= 1u && p.coord_system != 0u) {
        // Spherical / Polar / OriginSphere coordinate perturbation
        float noise = sample(cand_seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, ulong(leader), 0u);
        float inv_sqrt_d = rsqrt(max(float(p.dimensions), 1.0f));
        float radial = p.coord_system == 2u
            ? pow(max(unit_draw(cand_seed, leaf.key, 0ul, 9u), 1.0e-12f), 1.0f / max(float(p.dimensions), 1.0f))
            : 1.0f;
        float step = (leaf.scale * radius * half_scale) * radial * inv_sqrt_d * noise;
        if (p.coord_system == 3u) {
            float centered = base_value - 0.5f;
            proposed = 0.5f + cos(radius * 0.5f) * centered + sin(radius * 0.5f) * step;
        } else {
            proposed = base_value + step;
        }
        if (p.box_bounded != 0u) proposed = clamp(proposed, 0.0f, 1.0f);
    } else {
        float half_w = 0.5f * radius * half_scale;
        float tr_lo = radius >= 2.0f ? 0.0f : clamp(base_value - half_w, 0.0f, 1.0f);
        float tr_hi = radius >= 2.0f ? 1.0f : clamp(base_value + half_w, 0.0f, 1.0f);
        PoolKey rk = pool_key(stream_seed);
        if (mode == MODE_UNIFORM) {
            PoolKey kc = cand_key(rk, candidate);
            float u = hash_unit(kc, leader, 0x524141535056414cul);
            proposed = tr_hi > tr_lo ? clamp(tr_lo + u * (tr_hi - tr_lo), tr_lo, tr_hi) : tr_lo;
        } else if (mode == MODE_RAASP) {
            uint dims = max(p.dimensions, 1u);
            PoolKey kc = cand_key(rk, candidate);
            float prob = min(float(max(p.num_pert, 1u)) / float(dims), 1.0f);
            bool active = (prob >= 1.0f)
                || (leader == forced_dim(kc, dims))
                || (hash_unit(kc, leader, 0x52414153504d534bul) < prob);
            if (active) {
                float u = hash_unit(kc, leader, 0x524141535056414cul);
                proposed = tr_hi > tr_lo ? clamp(tr_lo + u * (tr_hi - tr_lo), tr_lo, tr_hi) : tr_lo;
            }
        } else if (mode == MODE_SOBOL || mode == MODE_MASKED) {
            uint dims = max(p.dimensions, 1u);
            bool active = true;
            if (mode == MODE_MASKED) {
                PoolKey kc = cand_key(rk, candidate);
                float prob = min(float(max(p.num_pert, 1u)) / float(dims), 1.0f);
                active = (prob >= 1.0f)
                    || (leader == forced_dim(kc, dims))
                    || (hash_unit(kc, leader, 0x52414153504d534bul) < prob);
            }
            if (active) {
                uint sobol_idx = p.sobol_offset + candidate;
                uint gray = sobol_idx ^ (sobol_idx >> 1u);
                uint raw = 0u;
                uint d_mod = leader % dims;
                for (uint b = 0u; gray != 0u && b < 32u; b++, gray >>= 1u) {
                    if ((gray & 1u) != 0u) raw ^= sobol_dirs[d_mod * 32u + b];
                }
                ulong sh = key_seed(key_fold(key_fold(rk, 0x534f424f4c4f574eul), ulong(leader)));
                uint scramble = uint(sh) ^ uint(sh >> 32u);
                uint owen = owen_scramble(raw, scramble);
                float u = float(owen >> 8u) * 0x1.0p-24f;
                proposed = tr_hi > tr_lo ? clamp(tr_lo + u * (tr_hi - tr_lo), tr_lo, tr_hi) : tr_lo;
            }
        } else if (mode == MODE_LHD) {
            uint n_pts = max(p.lhd_points, 1u);
            uint slot = candidate % n_pts;
            uint stratum = lhd_bin(rk, slot, n_pts, leader);
            float jitter = hash_unit(cand_key(rk, slot), leader, 0x4c48445354524154ul);
            proposed = clamp((float(stratum) + jitter) / float(n_pts), 0.0f, 1.0f);
        }
    }
    if (cats >= 2u) {
        float denom = float(cats - 1u);
        proposed = clamp(rint(proposed * denom) / denom, 0.0f, 1.0f);
    }
    return encode(proposed);
}

#ifdef FP16_INDEPENDENT
inline ulong proposal_seed(ulong root, uint candidate, uint pool_slots) {
    return candidate_seed(root, candidate, pool_slots);
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
    return (leaf.scale * radius) * noise;
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
        proposal_seed(p.stream_seed, candidate, p.pool_slots),
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
    device const DimMeta *dim_meta [[buffer(9)]],
    device const uint *sobol_dirs [[buffer(10)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float a = 0.0f, b = 0.0f, sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
    bool custom = (p.mode >= 2u) || (p.box_bounded != 0u) || (p.has_dim_meta != 0u) || (p.coord_system != 0u);
    for (uint item = tid; item < tile.length; item += 256) {
        ulong element = ulong(tile.start) + item;
        ulong index = leaf.offset + element;
        float base_value = decode(base[index]);
        ushort value;
        if (custom) {
            value = sample_manifold(
                base_value, leaf, p.seed, p.stream_seed, p.candidate, p.radius, p.mode, index, p, dim_meta, sobol_dirs);
        } else {
#ifdef FP16_INDEPENDENT
            ulong first = (element / 2ul) * 2ul;
            ushort2 base_pair = ushort2(base[leaf.offset + first],
                first + 1ul < leaf.length ? base[leaf.offset + first + 1ul] : 0);
            ushort2 values = proposal_pair(decode_pair(base_pair), leaf, p.seed,
                p.candidate, p.radius, p.mode,
                element / 2ul, p);
            value = values[element & 1ul];
#else
            float noise = sample(p.seed ^ 0x8ebc6af09c88c6e3ul, leaf.key, element, p.mode);
            value = encode(base_value + (leaf.scale * p.radius) * noise);
#endif
        }
        output[index] = value;
        invalid |= uint(invalid_value(value));
        changed += uint(value != base[index]);
        float w = p.has_dim_meta != 0u ? dim_meta[index].weight : leaf.weight;
        float value_float = decode(value);
        float delta = value_float - base_value;
        sq += delta * delta;
        norm += (delta * delta) * w;
        if (p.base_slot != 0) {
            delta = value_float - decode(anchor[index]);
            a += (delta * delta) * w;
        }
        if (p.history == 2 && p.base_slot != 1) {
            delta = value_float - decode(rejected[index]);
            b += (delta * delta) * w;
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
    device const DimMeta *dim_meta [[buffer(9)]],
    device const uint *sobol_dirs [[buffer(10)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    float sq = 0.0f, norm = 0.0f;
    uint changed = 0, invalid = 0;
    bool custom = (p.mode >= 2u) || (p.box_bounded != 0u) || (p.has_dim_meta != 0u) || (p.coord_system != 0u);
    if (custom) {
        for (uint item = tid; item < tile.length; item += 256) {
            ulong element = ulong(tile.start) + item;
            ulong index = leaf.offset + element;
            ushort base_bits = base[index];
            float base_value = decode(base_bits);
            ushort value = sample_manifold(
                base_value, leaf, p.seed, p.stream_seed, p.candidate, p.radius, p.mode, index, p, dim_meta, sobol_dirs);
            output[index] = value;
            invalid |= uint(invalid_value(value));
            changed += uint(value != base_bits);
            float w = p.has_dim_meta != 0u ? dim_meta[index].weight : leaf.weight;
            float delta = decode(value) - base_value;
            sq += delta * delta;
            norm += (delta * delta) * w;
        }
    } else {
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
    }
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

// Tiled over (tile_index, cand_group) where each cand_group handles up to 4
// candidates in O(1) registers and threadgroup memory for arbitrary pool_count!
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
    device const DimMeta *dim_meta [[buffer(9)]],
    device const uint *sobol_dirs [[buffer(10)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tile_index = tg_pos.x;
    uint cand_group = tg_pos.y;
    uint cand_base = cand_group * 4u;
    if (cand_base >= p.pool_count) return;
    uint group_count = min(4u, p.pool_count - cand_base);
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
    bool custom = (p.mode >= 2u) || (p.box_bounded != 0u) || (p.has_dim_meta != 0u) || (p.coord_system != 0u);
    ulong cand_seeds[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint c = 0; c < group_count; c++) {
        cand_seeds[c] = candidate_seed(p.stream_seed, cand_base + c, p.pool_slots);
    }

    for (uint item = tid * 2; item < tile.length; item += 512) {
        ulong pair_element = ulong(tile.start) + item;
        ulong pair = pair_element / 2;
        if (custom) {
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
                float w = p.has_dim_meta != 0u ? dim_meta[index].weight : leaf.weight;
                ushort values[4] = {base_bits, base_bits, base_bits, base_bits};
                for (uint c = 0; c < group_count; c++) {
                    values[c] = sample_manifold(
                        base_value, leaf, cand_seeds[c], p.stream_seed, cand_base + c, p.radius, p.mode, index, p, dim_meta, sobol_dirs);
                }
                if (p.pool_count == 1u && p.initialize == 0u) {
                    output[index] = values[0];
                }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                float deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
                for (uint c = 0; c < group_count; c++) {
                    ushort value = values[c];
                    invalid[c] |= uint(invalid_value(value));
                    changed[c] += uint(value != base_bits);
                    float value_float = decode(value);
                    float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                    deltas[c] = delta;
#endif
                    sq[c] += delta * delta;
                    norms[c] += (delta * delta) * w;
                    if (p.base_slot != 0) {
                        delta = value_float - anchor_value;
                        a[c] += (delta * delta) * w;
                    }
                    if (p.history == 2 && p.base_slot != 1) {
                        delta = value_float - rejected_value;
                        b[c] += (delta * delta) * w;
                    }
                }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                if (cand_group == 0u) {
                    dots[0] += deltas[0] * deltas[1] * w;
                    dots[1] += deltas[0] * deltas[2] * w;
                    dots[2] += deltas[0] * deltas[3] * w;
                    dots[3] += deltas[1] * deltas[2] * w;
                    dots[4] += deltas[1] * deltas[3] * w;
                    dots[5] += deltas[2] * deltas[3] * w;
                }
#endif
            }
        } else {
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
            for (uint c = 0; c < group_count; c++)
                values[c] = procedural_pair(
                    base_values,
                    leaf,
                    p,
                    pair,
                    cand_base + c);
            if (p.pool_count == 1u && p.initialize == 0u) {
                if (aligned_pair) {
                    *reinterpret_cast<device ushort2*>(output + first_index) = values[0];
                } else {
                    output[first_index] = values[0].x;
                    if (complete_pair) output[first_index + 1] = values[0].y;
                }
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
                for (uint c = 0; c < group_count; c++) {
                    ushort value = values[c][pair_offset];
                    invalid[c] |= uint(invalid_value(value));
                    changed[c] += uint(value != base_bits);
                    float value_float = decode(value);
                    float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                    deltas[c] = delta;
#endif
                    sq[c] += delta * delta;
                    norms[c] += (delta * delta) * leaf.weight;
                    if (p.base_slot != 0) {
                        delta = value_float - anchor_value;
                        a[c] += (delta * delta) * leaf.weight;
                    }
                    if (p.history == 2 && p.base_slot != 1) {
                        delta = value_float - rejected_value;
                        b[c] += (delta * delta) * leaf.weight;
                    }
                }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                if (cand_group == 0u) {
                    dots[0] += deltas[0] * deltas[1] * leaf.weight;
                    dots[1] += deltas[0] * deltas[2] * leaf.weight;
                    dots[2] += deltas[0] * deltas[3] * leaf.weight;
                    dots[3] += deltas[1] * deltas[2] * leaf.weight;
                    dots[4] += deltas[1] * deltas[3] * leaf.weight;
                    dots[5] += deltas[2] * deltas[3] * leaf.weight;
                }
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
                for (uint c = 0; c < group_count; c++) {
                    float2 samples = noise_pair(cand_seeds[c]
                        ^ 0x8ebc6af09c88c6e3ul, leaf.key, pair, p.mode);
                    float noise = samples[pair_offset];
                    values[c] = encode(base_value + radius * noise);
                }
                if (p.pool_count == 1u && p.initialize == 0u) {
                    output[index] = values[0];
                }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                float deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
                for (uint c = 0; c < group_count; c++) {
                    ushort value = values[c];
                    invalid[c] |= uint(invalid_value(value));
                    changed[c] += uint(value != base_bits);
                    float value_float = decode(value);
                    float delta = value_float - base_value;
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                    deltas[c] = delta;
#endif
                    sq[c] += delta * delta;
                    norms[c] += (delta * delta) * leaf.weight;
                    if (p.base_slot != 0) {
                        delta = value_float - anchor_value;
                        a[c] += (delta * delta) * leaf.weight;
                    }
                    if (p.history == 2 && p.base_slot != 1) {
                        delta = value_float - rejected_value;
                        b[c] += (delta * delta) * leaf.weight;
                    }
                }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
                if (cand_group == 0u) {
                    dots[0] += deltas[0] * deltas[1] * leaf.weight;
                    dots[1] += deltas[0] * deltas[2] * leaf.weight;
                    dots[2] += deltas[0] * deltas[3] * leaf.weight;
                    dots[3] += deltas[1] * deltas[2] * leaf.weight;
                    dots[4] += deltas[1] * deltas[3] * leaf.weight;
                    dots[5] += deltas[2] * deltas[3] * leaf.weight;
                }
#endif
            }
#endif
        }
    }
    if (p.base_slot == 0) {
        for (uint c = 0; c < group_count; c++) a[c] = norms[c];
    } else if (p.base_slot == 1) {
        for (uint c = 0; c < group_count; c++) b[c] = norms[c];
    }
    threadgroup float reduced_a[32], reduced_b[32], reduced_sq[32], reduced_norm[32];
    threadgroup uint reduced_counts[32], reduced_flags[32];
    uint lane = tid & 31u;
    uint simdgroup = tid >> 5;
    for (uint c = 0; c < group_count; c++) {
        float a_sum = simd_sum(a[c]);
        float b_sum = simd_sum(b[c]);
        float sq_sum = simd_sum(sq[c]);
        float norm_sum = simd_sum(norms[c]);
        uint changed_sum = simd_sum(changed[c]);
        uint invalid_sum = simd_sum(invalid[c]);
        if (lane == 0) {
            uint slot = c * 8 + simdgroup;
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
        for (uint c = 0; c < group_count; c++) {
            bool active = lane < 8;
            uint slot = c * 8 + min(lane, 7u);
            float a_sum = simd_sum(active ? reduced_a[slot] : 0.0f);
            float b_sum = simd_sum(active ? reduced_b[slot] : 0.0f);
            float sq_sum = simd_sum(active ? reduced_sq[slot] : 0.0f);
            float norm_sum = simd_sum(active ? reduced_norm[slot] : 0.0f);
            uint changed_sum = simd_sum(active ? reduced_counts[slot] : 0u);
            uint invalid_sum = simd_sum(active ? reduced_flags[slot] : 0u);
            if (lane != 0) continue;
            uint global_cand = cand_base + c;
            partials[(p.initialize * p.pool_count + global_cand) * p.tiles + tile_index] =
                {a_sum, b_sum, sq_sum, changed_sum, uint(invalid_sum != 0)};
#ifndef POOL_GEOMETRY_DIAGNOSTICS
            if (p.initialize == 0)
                geometry[global_cand * p.tiles + tile_index] = norm_sum;
#endif
        }
    }
#ifdef POOL_GEOMETRY_DIAGNOSTICS
    threadgroup float sums_a[1024];
    for (uint c = 0; c < 4; c++)
        sums_a[c * 256 + tid] = c < group_count ? norms[c] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint c = 0; c < 4; c++) {
                uint slot = c * 256 + tid;
                sums_a[slot] += sums_a[slot + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && p.initialize == 0) {
        for (uint c = 0; c < group_count; c++)
            geometry[(cand_base + c) * p.tiles + tile_index] = sums_a[c * 256];
    }
    if (cand_group == 0u) {
        threadgroup float sums_b[1024], sums_sq[512];
        for (uint pair_i = 0; pair_i < 4; pair_i++)
            sums_b[pair_i * 256 + tid] = dots[pair_i];
        for (uint pair_i = 0; pair_i < 2; pair_i++)
            sums_sq[pair_i * 256 + tid] = dots[pair_i + 4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride != 0; stride >>= 1) {
            if (tid < stride) {
                for (uint pair_i = 0; pair_i < 4; pair_i++) {
                    uint slot = pair_i * 256 + tid;
                    sums_b[slot] += sums_b[slot + stride];
                }
                for (uint pair_i = 0; pair_i < 2; pair_i++) {
                    uint slot = pair_i * 256 + tid;
                    sums_sq[slot] += sums_sq[slot + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0 && p.initialize == 0) {
            uint dot_base = max(p.pool_count, 4u);
            if (p.pool_count < 4u) {
                for (uint c = p.pool_count; c < 4u; c++)
                    geometry[c * p.tiles + tile_index] = 0.0f;
            }
            for (uint pair_i = 0; pair_i < 4; pair_i++)
                geometry[(dot_base + pair_i) * p.tiles + tile_index] = sums_b[pair_i * 256];
            for (uint pair_i = 0; pair_i < 2; pair_i++)
                geometry[(dot_base + 4u + pair_i) * p.tiles + tile_index] = sums_sq[pair_i * 256];
        }
    }
#endif
}

inline void warp_output(
    uint y_bounds_mode,
    float y_lower,
    float y_upper,
    float affine_a,
    float affine_b,
    float affine_c,
    thread float &mean,
    thread float &standard_error,
    thread float &epistemic_se) {
    if (affine_a != 0.0f || affine_b != 1.0f || affine_c != 1.0f) {
        mean = affine_a + affine_b * mean;
        standard_error *= affine_c;
        epistemic_se *= affine_c;
    }
    if (y_bounds_mode == 1u) {
        float ex = exp(clamp(mean, -40.0f, 40.0f));
        mean = y_lower + ex;
        standard_error *= ex;
        epistemic_se *= ex;
    } else if (y_bounds_mode == 2u) {
        float ex = exp(clamp(-mean, -40.0f, 40.0f));
        mean = y_upper - ex;
        standard_error *= ex;
        epistemic_se *= ex;
    } else if (y_bounds_mode == 3u) {
        float s = 1.0f / (1.0f + exp(clamp(-mean, -40.0f, 40.0f)));
        float span = y_upper - y_lower;
        float jac = abs(span * s * (1.0f - s));
        mean = y_lower + span * s;
        standard_error *= jac;
        epistemic_se *= jac;
    }
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
        float mean = q.outcomes[q.incumbent_index];
        float standard_error = sqrt(max(q.aleatoric_scale, 1.0e-12f)) * q.y_scale;
        float epistemic_se = standard_error;
        warp_output(
            q.y_bounds_mode, q.y_lower, q.y_upper,
            q.affine_a, q.affine_b, q.affine_c,
            mean, standard_error, epistemic_se);
        mean_out = mean;
        standard_error_out = standard_error;
        neighbors_out = 0;
        return mean_out;
    }
    float y_scale_sq = max(q.y_scale * q.y_scale, 1.0e-12f);
    float var_gate = q.noise_aware != 0u ? 1.0f : 0.0f;
    if (q.adaptive_neighbors != 0) {
        float precision_sum = 0.0f, noise_sum = 0.0f, best_risk = INFINITY;
        uint chosen = 1;
        for (uint position = 0; position < count; position++) {
            uint index = nearest[position];
            float yvar = var_gate * q.variances[index] / y_scale_sq;
            float noise = q.aleatoric_scale + yvar;
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
            + var_gate * q.variances[index] / y_scale_sq;
        float weight = 1.0f / max(variance, 1.0e-12f);
        sum += weight;
        value += weight * q.outcomes[index];
        reference = max(reference, weight);
    }
    float mean = value / max(sum, 1.0e-12f);
    float aleatoric = 0.0f;
    for (uint position = 0; position < count; position++) {
        uint index = nearest[position];
        float yvar = var_gate * q.variances[index] / y_scale_sq;
        float variance = 1.0e-9f
            + q.epistemic_scale * scaled_distances[index]
            + q.aleatoric_scale
            + yvar;
        float weight = (1.0f / max(variance, 1.0e-12f)) / max(sum, 1.0e-12f);
        aleatoric += weight * (q.aleatoric_scale + yvar);
    }
    float epistemic_var = 1.0f / max(sum, 1.0e-12f);
    float standard_error = sqrt(epistemic_var + aleatoric) * q.y_scale;
    float epistemic_se = sqrt(epistemic_var) * q.y_scale;
    warp_output(
        q.y_bounds_mode, q.y_lower, q.y_upper,
        q.affine_a, q.affine_b, q.affine_c,
        mean, standard_error, epistemic_se);
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
                + var_gate * q.variances[index] / y_scale_sq;
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

// Tiled over (pair, cand_group) in O(1) registers and threadgroup memory.
kernel void dense_reduce_pool(
    device const Partial *partials [[buffer(0)]],
    device const float *geometry [[buffer(1)]],
    device Partial *aggregates [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint pair = tg_pos.x;
    uint cand_group = tg_pos.y;
    uint cand_base = cand_group * 4u;
    if (cand_base >= q.pool_count) return;
    uint group_count = min(4u, q.pool_count - cand_base);

    float anchor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rejected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float norm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint changed[4] = {0, 0, 0, 0};
    uint invalid[4] = {0, 0, 0, 0};
    for (uint tile = tid; tile < q.tiles; tile += 256) {
        for (uint c = 0; c < group_count; c++) {
            uint global_cand = cand_base + c;
            Partial p = partials[(pair * q.pool_count + global_cand) * q.tiles + tile];
            anchor[c] += p.anchor;
            rejected[c] += p.rejected;
            changed[c] |= p.changed;
            invalid[c] |= p.invalid | uint(!isfinite(p.squared));
            if (pair == 0)
                norm[c] += geometry[global_cand * q.tiles + tile];
        }
    }
    threadgroup float anchor_sums[1024], rejected_sums[1024], norm_sums[1024];
    threadgroup uint changed_flags[1024], invalid_flags[1024];
    for (uint c = 0; c < group_count; c++) {
        uint slot = c * 256 + tid;
        anchor_sums[slot] = anchor[c];
        rejected_sums[slot] = rejected[c];
        norm_sums[slot] = norm[c];
        changed_flags[slot] = changed[c];
        invalid_flags[slot] = invalid[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            for (uint c = 0; c < group_count; c++) {
                uint slot = c * 256 + tid;
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
        for (uint c = 0; c < group_count; c++) {
            uint slot = c * 256;
            aggregates[pair * q.pool_count + cand_base + c] = {
                anchor_sums[slot],
                rejected_sums[slot],
                norm_sums[slot],
                changed_flags[slot],
                invalid_flags[slot]
            };
        }
    }
}

// Stage 1 parallel candidate evaluation: 1 GPU thread per candidate across 0..q.pool_count!
kernel void dense_score_candidates(
    device const Partial *aggregates [[buffer(0)]],
    device float *pool_distances [[buffer(1)]],
    device CandidateScore *candidate_scores [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    device const float *axis_candidates [[buffer(4)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= q.pool_count) return;
    float distances[BAUX_HISTORY];
    float proposal_norm = aggregates[candidate].squared;
    bool changed = false;
    bool invalid = false;
    for (uint row = 0; row < q.history; row++) {
        distances[row] = q.implicit_history != 0
            ? q.base_distances[row] + proposal_norm
            : 0.0f;
    }
    for (uint row = 0; row < q.resident_history; row += 2) {
        Partial aggregate = aggregates[(row / 2) * q.pool_count + candidate];
        float first = aggregate.anchor;
        float second = aggregate.rejected;
        changed = changed || aggregate.changed != 0;
        invalid = invalid || aggregate.invalid != 0;
        uint first_index = q.implicit_history != 0 ? q.resident_indices[row] : row;
        if (first_index < q.history) {
            distances[first_index] = first;
        }
        if (row + 1 < q.resident_history) {
            uint second_index = q.implicit_history != 0 ? q.resident_indices[row + 1] : row + 1;
            if (second_index < q.history) {
                distances[second_index] = second;
            }
        }
        invalid = invalid || !isfinite(first)
            || (row + 1 < q.resident_history && !isfinite(second));
    }
    if (q.axis_enabled != 0) {
        float axis_cand = axis_candidates[candidate];
        changed = changed || (axis_cand != q.axis_base);
        for (uint row = 0; row < q.history; row++) {
            float delta = axis_cand - q.axis_history[row];
            distances[row] += q.axis_weight * delta * delta;
        }
    }
    for (uint row = 0; row < q.history; row++) {
        pool_distances[candidate * q.history + row] = distances[row];
    }
    if (candidate < q.candidate_floor || invalid) {
        candidate_scores[candidate] = {NAN, NAN, NAN, 0u, uint(changed), 1u};
        return;
    }
    float mean = NAN;
    float standard_error = NAN;
    uint neighbors = 0u;
    float score = acquisition_score(distances, q, BAUX_HISTORY, mean, standard_error, neighbors);
    bool finite_ok = isfinite(score) && isfinite(mean) && isfinite(standard_error);
    candidate_scores[candidate] = {
        mean,
        standard_error,
        score,
        neighbors,
        uint(changed),
        uint(!finite_ok)
    };
}

// Stage 2 parallel 256-thread argmax / Pareto / multi-arm reduction!
kernel void dense_select(
    device const CandidateScore *scores [[buffer(0)]],
    device Decision *decision [[buffer(1)]],
    device uint *selected_indices [[buffer(2)]],
    constant SelectionParams &q [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    // 1. Parallel reduction for `any_changed`
    uint local_changed = 0u;
    for (uint c = tid; c < q.pool_count; c += 256u) {
        if (scores[c].invalid == 0u && scores[c].changed != 0u) local_changed = 1u;
    }
    threadgroup uint shared_changed[256];
    shared_changed[tid] = local_changed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) shared_changed[tid] |= shared_changed[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool any_changed = shared_changed[0] != 0u;

    // 2. Incumbent prediction on tid == 0 while threads prepare reduction
    threadgroup float inc_mean_sh;
    threadgroup float inc_se_sh;
    threadgroup uint inc_nb_sh;
    if (tid == 0u) {
        float incumbent_distances[BAUX_HISTORY];
        for (uint row = 0; row < q.history; row++) {
            incumbent_distances[row] = q.base_distances[row]
                + (q.axis_enabled != 0
                    ? q.axis_weight * (q.axis_base - q.axis_history[row])
                        * (q.axis_base - q.axis_history[row])
                    : 0.0f);
        }
        float im = 0.0f, ise = 0.0f;
        uint inb = 0u;
        acquisition_score(incumbent_distances, q, BAUX_HISTORY, im, ise, inb);
        inc_mean_sh = im;
        inc_se_sh = ise;
        inc_nb_sh = inb;
    }

    uint arms_to_pick = max(min(q.num_arms, q.pool_count), 1u);
    threadgroup float red_score[256];
    threadgroup ulong red_prio[256];
    threadgroup uint red_idx[256];
    threadgroup uint picked_count;
    if (tid == 0u) picked_count = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q.forced_candidate != 0xffffffffu) {
        if (tid == 0u) {
            uint fc = q.forced_candidate;
            bool ok = fc < q.pool_count
                && fc >= q.candidate_floor
                && scores[fc].invalid == 0u
                && (!any_changed || scores[fc].changed != 0u);
            if (ok) {
                selected_indices[0] = fc;
                picked_count = 1u;
            } else {
                selected_indices[0] = 0xffffffffu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (q.acquisition == 2u) {
        // Parallel 2D (mean, se) Pareto frontier + priority tie-break across 0..q.pool_count
        for (uint arm = 0u; arm < arms_to_pick; arm++) {
            ulong best_prio = 0ul;
            uint best_c = 0xffffffffu;
            for (uint c = tid; c < q.pool_count; c += 256u) {
                CandidateScore cs = scores[c];
                if (c < q.candidate_floor || cs.invalid != 0u || (any_changed && cs.changed == 0u)) continue;
                bool already = false;
                for (uint prev = 0u; prev < arm; prev++) {
                    if (selected_indices[prev] == c) { already = true; break; }
                }
                if (already) continue;
                bool dominated = false;
                for (uint other = 0u; other < q.pool_count; other++) {
                    if (other == c) continue;
                    CandidateScore os = scores[other];
                    if (other < q.candidate_floor || os.invalid != 0u || (any_changed && os.changed == 0u)) continue;
                    bool prev_picked = false;
                    for (uint prev = 0u; prev < arm; prev++) {
                        if (selected_indices[prev] == other) { prev_picked = true; break; }
                    }
                    if (prev_picked) continue;
                    bool no_worse = os.mean >= cs.mean && os.standard_error >= cs.standard_error;
                    bool strictly_better = os.mean > cs.mean || os.standard_error > cs.standard_error;
                    if (no_worse && strictly_better) { dominated = true; break; }
                }
                if (!dominated) {
                    ulong prio = mix64(q.root_seed ^ (ulong(c) * 0x9e3779b97f4a7c15ul));
                    if (best_c == 0xffffffffu || prio > best_prio || (prio == best_prio && c < best_c)) {
                        best_prio = prio;
                        best_c = c;
                    }
                }
            }
            red_prio[tid] = best_prio;
            red_idx[tid] = best_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride != 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint rc = red_idx[tid + stride];
                    ulong rp = red_prio[tid + stride];
                    uint lc = red_idx[tid];
                    ulong lp = red_prio[tid];
                    if (rc != 0xffffffffu && (lc == 0xffffffffu || rp > lp || (rp == lp && rc < lc))) {
                        red_idx[tid] = rc;
                        red_prio[tid] = rp;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                selected_indices[arm] = red_idx[0];
                if (red_idx[0] != 0xffffffffu) picked_count = arm + 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (selected_indices[arm] == 0xffffffffu) break;
        }
    } else {
        // Parallel scalar argmax across 0..q.pool_count (picking up to `arms_to_pick` distinct winners)
        for (uint arm = 0u; arm < arms_to_pick; arm++) {
            float best_s = -INFINITY;
            uint best_c = 0xffffffffu;
            for (uint c = tid; c < q.pool_count; c += 256u) {
                CandidateScore cs = scores[c];
                if (c < q.candidate_floor || cs.invalid != 0u || (any_changed && cs.changed == 0u)) continue;
                bool already = false;
                for (uint prev = 0u; prev < arm; prev++) {
                    if (selected_indices[prev] == c) { already = true; break; }
                }
                if (already) continue;
                if (best_c == 0xffffffffu || cs.score > best_s || (cs.score == best_s && c < best_c)) {
                    best_s = cs.score;
                    best_c = c;
                }
            }
            red_score[tid] = best_s;
            red_idx[tid] = best_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride != 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint rc = red_idx[tid + stride];
                    float rs = red_score[tid + stride];
                    uint lc = red_idx[tid];
                    float ls = red_score[tid];
                    if (rc != 0xffffffffu && (lc == 0xffffffffu || rs > ls || (rs == ls && rc < lc))) {
                        red_idx[tid] = rc;
                        red_score[tid] = rs;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                selected_indices[arm] = red_idx[0];
                if (red_idx[0] != 0xffffffffu) picked_count = arm + 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (selected_indices[arm] == 0xffffffffu) break;
        }
    }

    if (tid == 0u) {
        uint selected = picked_count > 0u ? selected_indices[0] : 0xffffffffu;
        bool valid = selected < q.pool_count;
        CandidateScore win = valid ? scores[selected] : CandidateScore{0.0f, 0.0f, 0.0f, 0u, 0u, 1u};
        float reported_score = valid
            ? (q.forced_candidate != 0xffffffffu ? 0.0f : win.score)
            : 0.0f;
        decision->index = selected;
        decision->valid = valid ? 1u : 0u;
        decision->root_seed = q.root_seed;
        decision->seed = valid ? candidate_seed(q.root_seed, selected, q.pool_slots) : 0ul;
        decision->radius = valid ? q.radius : 0.0f;
        decision->score = reported_score;
        decision->mode = q.mode;
        decision->num_selected = picked_count;
        decision->predicted_mean = valid ? win.mean : 0.0f;
        decision->predicted_standard_error = valid ? win.standard_error : 0.0f;
        decision->incumbent_mean = inc_mean_sh;
        decision->incumbent_standard_error = inc_se_sh;
        decision->neighbors = valid ? win.neighbors : 0u;
        decision->incumbent_neighbors = inc_nb_sh;
        for (uint c = 0u; c < 4u; c++) {
            if (c < q.pool_count && scores[c].invalid == 0u && (!any_changed || scores[c].changed != 0u) && c >= q.candidate_floor) {
                decision->candidate_means[c] = scores[c].mean;
                decision->candidate_standard_errors[c] = scores[c].standard_error;
                decision->candidate_scores[c] = scores[c].score;
            } else {
                decision->candidate_means[c] = NAN;
                decision->candidate_standard_errors[c] = NAN;
                decision->candidate_scores[c] = NAN;
            }
        }
    }
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
    device const DimMeta *dim_meta [[buffer(8)]],
    device const uint *sobol_dirs [[buffer(9)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (decision->valid == 0) return;
    uint chosen_index = p.candidate != 0xffffffffu ? p.candidate : decision->index;
    ulong chosen_seed = p.candidate != 0xffffffffu
        ? candidate_seed(decision->root_seed, chosen_index, p.pool_slots)
        : decision->seed;
    float chosen_radius = decision->radius;
    uint chosen_mode = decision->mode;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    bool custom = (chosen_mode >= 2u) || (p.box_bounded != 0u) || (p.has_dim_meta != 0u) || (p.coord_system != 0u);
#ifdef FP16_INDEPENDENT
    ulong2 domain = ziggurat_domain(chosen_seed ^ 0x8ebc6af09c88c6e3ul, leaf.key);
#else
    ulong seed = chosen_seed ^ 0x8ebc6af09c88c6e3ul;
    const float step = leaf.scale * chosen_radius;
#endif
    float squared = 0.0f;
    uint changed = 0u;
    uint invalid = 0u;
    if (custom) {
        for (uint item = tid; item < tile.length; item += 256) {
            ulong element = ulong(tile.start) + item;
            ulong index = leaf.offset + element;
            ushort base_bits = base[index];
            float base_value = decode(base_bits);
            ushort value = sample_manifold(
                base_value, leaf, chosen_seed, decision->root_seed, chosen_index, chosen_radius, chosen_mode, index, p, dim_meta, sobol_dirs);
            output[index] = value;
            if (account != 0u) {
                changed += uint(value != base_bits);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - base_value;
                squared += delta * delta;
            }
        }
    } else {
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
                chosen_index,
                chosen_radius, chosen_mode, pair);
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
            float2 samples = noise_pair(seed, leaf.key, pair, chosen_mode);

            ushort first_bits = base[index];
            output[index] = encode(decode(first_bits) + step * samples.x);
            if (account != 0u) {
                ushort value = output[index];
                changed += uint(value != first_bits);
                invalid |= uint(invalid_value(value));
                float delta = decode(value) - decode(first_bits);
                squared += delta * delta;
            }

            if (item + 1 < tile.length) {
                ushort second_bits = base[index + 1];
                output[index + 1] = encode(decode(second_bits) + step * samples.y);
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
            partials[chosen_index * p.tiles + tile_index] = {
                0.0f, 0.0f, total_squared, total_changed,
                uint(total_invalid != 0u)
            };
    }
}
