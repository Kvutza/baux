typedef struct {
    ulong seed;
    float coefficient;
    uint pad;
} BauxDenseTerm;

inline ulong baux_dense_mix64(ulong input) {
    ulong value = input + 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

inline float baux_dense_sign(ulong seed, ulong leaf_key, ulong element) {
    ulong leaf = baux_dense_mix64(leaf_key ^ 0xd6e8feb86659fd93ul);
    ulong coordinate = baux_dense_mix64(element ^ 0xa0761d6478bd642ful);
    return (baux_dense_mix64(seed ^ leaf ^ coordinate) & 1ul) == 0ul
        ? -1.0f
        : 1.0f;
}

inline float baux_dense_next(float value, int positive) {
    uint bits = as_uint(value);
    if (value == 0.0f) {
        return as_float(positive ? 1u : 0x80000001u);
    }
    bits = ((value > 0.0f) == positive) ? bits + 1u : bits - 1u;
    float candidate = as_float(bits);
    if (isfinite(candidate)) {
        return candidate;
    }
    bits = as_uint(value);
    bits = ((value > 0.0f) == positive) ? bits - 1u : bits + 1u;
    return as_float(bits);
}

inline float baux_dense_value(
    float value,
    ulong leaf_key,
    ulong element,
    float scale,
    __global const BauxDenseTerm* terms,
    uint term_count
) {
    float sum = 0.0f;
    float strongest = 0.0f;
    int positive = 1;
    for (uint term_index = 0; term_index < term_count; ++term_index) {
        BauxDenseTerm term = terms[term_index];
        if (term.coefficient == 0.0f) {
            continue;
        }
        float direction = baux_dense_sign(term.seed, leaf_key, element);
        sum += term.coefficient * direction;
        if (fabs(term.coefficient) > strongest) {
            strongest = fabs(term.coefficient);
            positive = (term.coefficient > 0.0f) == (direction > 0.0f);
        }
    }
    float candidate = value + scale * sum;
    return (sum == 0.0f || candidate == value)
        ? baux_dense_next(value, positive)
        : candidate;
}
