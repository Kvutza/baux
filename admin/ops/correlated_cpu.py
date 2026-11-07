"""CPU numerical reference helpers for correlated and Gaussian BF16 parity."""

import math
import struct

import numpy as np

MASK64, EPS = (1 << 64) - 1, np.finfo(np.float32).eps
NOISE_DOMAIN, REFERENCE_DOMAIN = 0x8EBC6AF09C88C6E3, 0xE7037ED1A0B428DB


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def decode(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def encode(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def trial_hash(low, high, index):
    val = low ^ ((index * 0x9E3779B9) & 0xFFFFFFFF)
    val = (((val ^ (val >> 16)) * 0x7FEB352D) & 0xFFFFFFFF) ^ high
    val = (val * 0x846CA68B) & 0xFFFFFFFF
    return val ^ (val >> 15)


def candidate_seed(root, stream):
    low, high = root & 0xFFFFFFFF, root >> 32
    return trial_hash(low, high, stream) | (
        trial_hash(high, low, stream ^ 0x9E3779B9) << 32
    )


def dense_normal(seed, key, element):
    """Mirror FP32 Box-Muller operations with host libm, not CUDA libdevice."""
    first = mix64(
        seed
        ^ mix64(key ^ 0xD6E8FEB86659FD93)
        ^ mix64((element // 2) ^ 0xA0761D6478BD642F)
    )
    u1 = f32(((first >> 11) + 1) / (1 << 53))
    u2 = f32((mix64(first ^ 0xD2B74407B1CE6E93) >> 11) / (1 << 53))
    log_u1 = f32(math.log(min(max(u1, f32(1e-12)), f32(0.99999994))))
    radius = f32(math.sqrt(f32(-2.0 * log_u1)))
    trig = math.cos if element % 2 == 0 else math.sin
    return f32(radius * f32(trig(f32(f32(math.tau) * u2))))


def draw_normal(seed, slot):
    combined = ((seed * 1_000_003 + slot) * 1_000_003) & MASK64
    u1 = min(max((mix64(combined) >> 11) / (1 << 53), 1e-12), 1 - 1e-12)
    u2 = (mix64(combined ^ 0xD2B74407B1CE6E93) >> 11) / (1 << 53)
    return f32(math.sqrt(-2 * math.log(u1)) * math.cos(math.tau * u2))


def initial_reference(leaves, seed):
    return np.asarray(
        [
            dense_normal(seed ^ REFERENCE_DOMAIN, key, i)
            for key, _, length, _ in leaves
            for i in range(length)
        ],
        dtype=np.float32,
    )


def _leafdir(leaf, reference, seed, persistence):
    key, off, ln, _ = leaf
    noises = [dense_normal(seed ^ NOISE_DOMAIN, key, i) for i in range(ln)]
    if not persistence:
        return noises, [16 * EPS * (1 + abs(n)) for n in noises]
    stored = decode(reference[off : off + ln]).astype(np.float64)
    inv_rms = f32(math.sqrt(ln / np.square(stored).sum()))
    lefts = [f32(persistence * f32(float(s) * inv_rms)) for s in stored]
    rights = [f32(f32(math.sqrt(0.4375)) * n) for n in noises]
    return (
        [f32(lv + rv) for lv, rv in zip(lefts, rights)],
        [16 * EPS * (1 + abs(lv) + abs(rv)) for lv, rv in zip(lefts, rights)],
    )


def direction(leaves, reference, seed, persistence):
    parts = [_leafdir(leaf, reference, seed, persistence) for leaf in leaves]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def candidate(base, leaves, reference, root, index, radius, *, correlated=True):
    seed = candidate_seed(root, index // 2 if correlated else index)
    values, errors = direction(
        leaves, reference, seed, 0.75 if correlated and index < 2 else 0.0
    )
    old = decode(base).astype(np.float64)
    raw, allowance = np.empty(len(base)), np.empty(len(base))
    for _, off, ln, scale in leaves:
        sl, step = slice(off, off + ln), f32(f32(scale) * f32(radius))
        delta = (step * values[sl]).astype(np.float32).astype(np.float64)
        raw[sl] = (old[sl] + delta).astype(np.float32)
        allowance[sl] = abs(step) * errors[sl] + 4 * EPS * (abs(old[sl]) + abs(delta))
    return tuple([seed, values, errors, raw, allowance])


def rounded_bounds(raw, allowance):
    raw, allowance = np.asarray(raw), np.asarray(allowance)
    return (
        decode(encode(raw - allowance)).astype(np.float64),
        decode(encode(raw + allowance)).astype(np.float64),
    )


def check_rounding(actual, raw, allowance):
    """Permit only adjacent BF16 rounding cells crossed by a tiny FP32 interval."""
    actual, expected = np.asarray(actual, dtype=np.uint16), encode(raw)
    low, high = rounded_bounds(raw, allowance)
    decoded = decode(actual).astype(np.float64)
    assert actual.shape == expected.shape and np.isfinite(decoded).all()
    assert np.all((decoded >= low) & (decoded <= high)), (
        "Outside FP32 rounding-boundary allowance"
    )
    mismatch = actual != expected
    for obs, want in zip(actual[mismatch], expected[mismatch]):
        assert (int(obs) ^ int(want)) & 0x8000 == 0 and abs(int(obs) - int(want)) == 1
    return int(np.count_nonzero(mismatch))


def score(bits, history, draw_seed):
    values = decode(bits).astype(np.float64)
    dists = [np.square(values - row).sum() for _, row, _ in history]
    weights = 1 / (1e-9 + 0.7 * np.asarray(dists) + 0.05)
    rewards = np.asarray([reward for _, _, reward in history])
    noises = np.asarray([draw_normal(draw_seed, slot) for slot, _, _ in history])
    return np.dot(weights, rewards) / weights.sum() + (
        np.dot(weights, noises) / np.linalg.norm(weights) / np.sqrt(weights.sum())
    )


def _ratio(wlo, whi, values, d_lo, d_hi):
    a, b = wlo * values, whi * values
    corners = [
        n / d
        for n in (np.minimum(a, b).sum(), np.maximum(a, b).sum())
        for d in (d_lo, d_hi)
    ]
    return min(corners), max(corners)


def score_bounds(raw, allowance, history, draw_seed):
    """Conservative TS interval when a CPU value straddles a BF16 midpoint."""
    low, high = rounded_bounds(raw, allowance)
    min_d = [
        np.square(np.maximum(np.maximum(low - row, row - high), 0)).sum()
        for _, row, _ in history
    ]
    max_d = [
        np.square(np.maximum(abs(low - row), abs(high - row))).sum()
        for _, row, _ in history
    ]
    wlo = 1 / (1e-9 + 0.7 * np.asarray(max_d) + 0.05)
    whi = 1 / (1e-9 + 0.7 * np.asarray(min_d) + 0.05)
    mean = _ratio(
        wlo, whi, np.asarray([r for _, _, r in history]), wlo.sum(), whi.sum()
    )
    noises = np.asarray([draw_normal(draw_seed, slot) for slot, _, _ in history])
    noise = _ratio(
        wlo,
        whi,
        noises,
        np.linalg.norm(wlo) * np.sqrt(wlo.sum()),
        np.linalg.norm(whi) * np.sqrt(whi.sum()),
    )
    return mean[0] + noise[0], mean[1] + noise[1]


def _isnull(item, base_dec):
    low, high = rounded_bounds(item[3], item[4])
    return np.array_equal(low, base_dec) and np.array_equal(high, base_dec)


def _ischanged(item, base_dec):
    low, high = rounded_bounds(item[3], item[4])
    return bool(np.any((low > base_dec) | (high < base_dec)))


def null_case():
    leaves, base, radius = [(17, 0, 33, 1.0)], encode(np.full(33, 1.5)), 0.0015
    ref, base_dec = encode(initial_reference(leaves, 123)), decode(base)
    for root in range(128):
        pool = [
            candidate(base, leaves, ref, root, i, radius * (0.5 if i % 2 == 0 else 2))
            for i in range(4)
        ]
        baseline = [
            candidate(base, leaves, None, root, i, radius, correlated=False)
            for i in range(4)
        ]
        if (
            _isnull(pool[0], base_dec)
            and _isnull(pool[2], base_dec)
            and _ischanged(pool[1], base_dec)
            and _isnull(baseline[0], base_dec)
            and any(_ischanged(it, base_dec) for it in baseline[1:])
        ):
            return leaves, base, radius, root
    raise AssertionError("No rounding-safe null/non-null fixture found")


def _checkmidpoints():
    for raw, neighbor in ((1.0, 0x3F81), (-1.0, 0xBF81)):
        assert check_rounding(encode([raw]), [raw], [1e-7]) == 0
        try:
            check_rounding([neighbor], [raw], [1e-7])
        except AssertionError:
            pass
        else:
            raise AssertionError("Tolerance accepted a BF16 ULP away from a midpoint")
    midpoint = 1 + 1 / 256
    assert check_rounding([0x3F81], [midpoint], [1e-7]) == 1
    assert check_rounding([0xBF81], [-midpoint], [1e-7]) == 1


def check_cpuhelpers():
    null_case()
    values = np.asarray([dense_normal(123, 17, i) for i in range(16_384)])
    assert abs(values.mean()) < 0.03 and abs(values.var() - 1) < 0.04
    assert np.unique(encode(values)).size > 1000
    _checkmidpoints()
    leaves = [(17, 0, 33, 1.0), (19, 33, 17, 0.5)]
    ref, base = encode(initial_reference(leaves, 123)), encode(np.ones(50))
    near = [candidate(base, leaves, ref, 456, i, 0.1) for i in range(4)]
    for a, b in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(near[a][1], near[b][1])
    independent = candidate(base, leaves, None, 456, 1, 0.1, correlated=False)
    assert independent[0] == candidate_seed(456, 1)
    history = [(1, decode(base).astype(np.float64), -0.5), (2, np.zeros(50), 0.75)]
    for _, _, _, raw, allowance in near:
        lo, hi = score_bounds(raw, allowance, history, 789)
        assert lo - 1e-12 <= score(encode(raw), history, 789) <= hi + 1e-12
