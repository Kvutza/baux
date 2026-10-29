#define BAUX_BF16_THREADS 256u

typedef struct {
    ulong key;
    ulong offset;
    ulong len;
    float scale;
    uint pad;
} BauxBf16Leaf;

typedef struct {
    uint leaf;
    uint start;
    uint len;
    uint pad;
} BauxBf16Tile;

inline float baux_bf16_decode(ushort value) {
    return as_float((uint)value << 16);
}

inline ushort baux_bf16_encode(float value) {
    uint bits = as_uint(value);
    return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline ushort baux_bf16_next(ushort value, int positive) {
    if ((value & 0x7fffu) == 0u) {
        return positive ? (ushort)1u : (ushort)0x8001u;
    }
    int grows = (((value & 0x8000u) == 0u) == positive);
    ushort candidate = grows ? (ushort)(value + 1u) : (ushort)(value - 1u);
    if (isfinite(baux_bf16_decode(candidate))) {
        return candidate;
    }
    return grows ? (ushort)(value - 1u) : (ushort)(value + 1u);
}

__kernel void materialize_bf16(
    __global const ushort* base,
    __global const BauxBf16Leaf* leaves,
    __global const BauxBf16Tile* tiles,
    __global ushort* candidate,
    __global const BauxDenseTerm* terms,
    uint term_count
) {
    uint thread_index = get_local_id(0);
    uint group_index = get_group_id(0);
    BauxBf16Tile tile = tiles[group_index];
    BauxBf16Leaf leaf = leaves[tile.leaf];
    for (uint item = thread_index; item < tile.len; item += BAUX_BF16_THREADS) {
        ulong coordinate = (ulong)(tile.start + item);
        float sum = 0.0f;
        float strongest = 0.0f;
        int positive = 1;
        for (uint term = 0; term < term_count; ++term) {
            float coefficient = terms[term].coefficient;
            if (coefficient == 0.0f) {
                continue;
            }
            float direction = baux_dense_sign(terms[term].seed, leaf.key, coordinate);
            sum = fma(coefficient, direction, sum);
            if (fabs(coefficient) > strongest) {
                strongest = fabs(coefficient);
                positive = ((coefficient > 0.0f) == (direction > 0.0f));
            }
        }
        ulong index = leaf.offset + coordinate;
        ushort value = baux_bf16_encode(
            baux_bf16_decode(base[index]) + leaf.scale * sum
        );
        candidate[index] = (sum == 0.0f || value == base[index])
            ? baux_bf16_next(base[index], positive)
            : value;
    }
}
