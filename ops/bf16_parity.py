"""Verify CUDA BF16 perturbations bit-for-bit against an independent reference."""

import gc
import math
import struct
from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from numpy import array_equal as arr_eq, concatenate as cat
from numpy.testing import assert_allclose as allclose

from baux._rust import ENN
from baux.experimental import ParamBlock, ParameterBuffer, turbo_enn

MASK64, MAX_F32, F32, F64 = (1 << 64) - 1, 3.4028234663852886e38, np.float32, np.float64


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def sign(seed: int, key: int, element: int) -> float:
    h = mix64(
        seed ^ mix64(key ^ 0xD6E8FEB86659FD93) ^ mix64(element ^ 0xA0761D6478BD642F)
    )
    return -1.0 if h & 1 == 0 else 1.0


def draw_normal(seed: int, index: int, metric: int) -> float:
    combined = ((((seed & MASK64) * 1_000_003 + index) * 1_000_003) + metric) & MASK64
    u1 = min(max((mix64(combined) >> 11) / float(1 << 53), 1.0e-12), 1.0 - 1.0e-12)
    u2 = (mix64(combined ^ 0xD2B74407B1CE6E93) >> 11) / float(1 << 53)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _drawpt(s, q, m, ctx):
    idx, w, l2, mu, err = ctx
    dot = sum(
        float(w[q, n, m]) * draw_normal(s, int(i), m) for n, i in enumerate(idx[q])
    )
    return float(mu[q, m]) + (float(err[q, m]) / max(float(l2[q, m]), 1.0e-12)) * dot


def draw_expected(indices, weights, l2, means, errors, seeds):
    ctx, q_r, m_r = (
        (indices, weights, l2, means, errors),
        range(means.shape[0]),
        range(means.shape[1]),
    )
    return np.array(
        [[[_drawpt(s, q, m, ctx) for m in m_r] for q in q_r] for s in seeds],
        dtype=np.float64,
    )


def decode(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def encode(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16


def next_value(value: int, positive: bool) -> int:
    if value & 0x7FFF == 0:
        return 1 if positive else 0x8001
    step, cand = (
        (s := 1 if (value & 0x8000 == 0) == positive else -1),
        (value + s) & 0xFFFF,
    )
    return cand if math.isfinite(decode(cand)) else (value - step) & 0xFFFF


def _cpuelem(raw_bit: int, key: int, local: int, scale: float, terms) -> int:
    active = [(c, sign(s, key, local)) for s, c in terms if c != 0.0]
    total = reduce(lambda acc, cd: f32(acc + f32(cd[0] * cd[1])), active, 0.0)
    best_c, best_dir = max(active, key=lambda cd: abs(cd[0]), default=(1.0, 1.0))
    cand, step_up = (
        encode(f32(decode(raw_bit) + f32(scale * total))),
        (best_c > 0.0) == (best_dir > 0.0),
    )
    return next_value(raw_bit, step_up) if total == 0.0 or cand == raw_bit else cand


def cpu_values(base, leaves, terms):
    return [
        _cpuelem(base[off + i], key, i, sc, terms)
        for key, off, ln, sc in leaves
        for i in range(ln)
    ]


def cpu_seedvalues(base, leaves, seed, radius):
    return [
        encode(
            f32(
                decode(base[off + i])
                + f32(f32(f32(sc) * f32(radius)) * sign(seed, k, i))
            )
        )
        for k, off, ln, sc in leaves
        for i in range(ln)
    ]


def _uints(arr):
    return np.array(
        jax.device_get(jax.lax.bitcast_convert_type(arr, jnp.uint16)), copy=True
    )


def check_descriptions(proposals, batch, base_bits, leaves, radius):
    actual = _uints(batch)
    assert len(proposals.describe()) == proposals.arms and actual.shape == (
        proposals.arms,
        len(base_bits),
    )
    for arm, (seed, score, obs_r, changes) in enumerate(proposals.describe()):
        exp = cpu_seedvalues(base_bits, leaves, seed, radius)
        assert (
            0 <= seed <= MASK64
            and math.isfinite(score)
            and obs_r == f32(radius)
            and len(changes) == len(leaves)
            and np.array_equal(actual[arm], exp)
        )
        for (_, off, ln, _), (changed, squared) in zip(leaves, changes):
            deltas = [
                f32(decode(exp[i]) - decode(base_bits[i])) for i in range(off, off + ln)
            ]
            sq_sum = reduce(lambda a, d: f32(a + f32(d * d)), deltas, 0.0)
            assert (
                changed == sum(exp[i] != base_bits[i] for i in range(off, off + ln))
                and np.testing.assert_allclose(squared, sq_sum, rtol=3.0e-6, atol=0.0)
                is None
            )
    return actual


def _fails(action, text: str) -> None:
    try:
        action()
    except ValueError as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def _checktol(search, accepted, leaves) -> None:
    for seed, exp_len in zip((47, 53, 59, 61), (0.5, 0.25, 0.25, 0.125)):
        prop = search.ask(1, 4, 1, seed)
        check_descriptions(
            prop, jax.dlpack.from_dlpack(prop), accepted.tolist(), leaves, search.length
        )
        gc.collect()
        search.tell(prop, [0.0])
        assert (search.sync(), search.length, search.best) == (
            [False],
            exp_len,
            1.0,
        ) and np.array_equal(search.read_best(), accepted)
    assert (search.restarts, search.history_len) == (0, 6)
    prop, batch = (p := search.ask(1, 4, 1, 67)), jax.dlpack.from_dlpack(p)
    exp = check_descriptions(prop, batch, accepted.tolist(), leaves, 0.125)
    del prop, search
    gc.collect()
    assert np.array_equal(_uints(batch), exp)
    del batch
    gc.collect()


def check_singlesearch():
    leaves = [(17, 0, 257, 2.0**-20), (19, 257, 516, 2.0**-7), (23, 773, 257, 0.25)]
    bits = ([0x3FC0, 0x3FC1, 0xBFC0, 0xBFC1] * 258)[:1030]
    base = jax.device_put(jnp.array(bits, dtype=jnp.uint16)).view(jnp.bfloat16)
    search = turbo_enn(
        base,
        -0.75,
        [ParamBlock(*lf) for lf in leaves],
        8,
        length_init=0.5,
        failure_tolerance=2,
    )
    init, prop = search.read_best(), search.ask(1, 4, 1, 0x123456789ABCDEF0)
    batch = jax.dlpack.from_dlpack(prop)
    batch.block_until_ready()
    acc = check_descriptions(prop, batch, bits, leaves, 0.5)[0].copy()
    assert (
        (init.dtype, prop.arms, batch.dtype) == (np.uint16, 1, jnp.bfloat16)
        and arr_eq(init, bits)
        and arr_eq(acc[:257], bits[:257])
        and np.all((acc[257:773] & 1) == 0)
        and np.any(acc[257:773] != bits[257:773])
        and np.all(acc[773:] != bits[773:])
    )
    for op in (
        partial(search.tell, prop, [1.0]),
        partial(search.ask, 1, 4, 1, 43),
        search.read_best,
    ):
        _fails(op, "live JAX proposals")
    del batch
    gc.collect()
    _fails(search.read_best, "pending BF16 round")
    search.tell(prop, [1.0])
    assert (
        (search.sync(), search.best) == ([True], 1.0)
        and arr_eq(search.read_best(), acc)
        and arr_eq(init, bits)
    )
    _checktol(search, acc, leaves)


class LegacyExport:
    def __init__(self, proposals):
        self.proposals = proposals

    def __dlpack_device__(self):
        return self.proposals.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        return self.proposals.__dlpack__(stream=stream)


def check_export(sampler="independent"):
    import torch

    for value, delayed in ((0.5, False), (float("nan"), False), (0.5, True)):
        search = turbo_enn(
            jnp.ones(16, dtype=jnp.bfloat16),
            0.0,
            [ParamBlock(17, 0, 16, 1.0)],
            2,
            sampler=sampler,
        )
        prop, s, c = search.ask(1, 4, 1, 123), torch.cuda.Stream(), torch.cuda.Event()
        with torch.cuda.stream(s):
            tensor = torch.from_dlpack(LegacyExport(prop))
            if delayed:
                torch.cuda._sleep(2_000_000_000)
            tensor.fill_(value)
            c.record()
        if not delayed:
            s.synchronize()
        del tensor
        gc.collect()
        assert ((not c.query()) if delayed else True) and (
            search.tell(prop, [1.0]) or c.query()
        )
        _fails(search.sync, "proposal or rewards are invalid")


def _searchnext(search):
    prop = search.ask(2, 4, 1, 5, beta=1.3, draw_seed=29)
    search.tell(
        prop,
        jax.device_put(jnp.array([1.5, 0.25], F32)),
        jax.device_put(jnp.array([0.02, 0.03], F32)),
    )
    assert (search.sync(), search.best, search.history_len) == (
        [True, False],
        1.5,
        5,
    ) and math.isclose(search.best_variance, 0.02, rel_tol=1.0e-6)
    prop = search.ask(2, 4, 1, 7, beta=1.3, draw_seed=31)
    search.tell(prop, jax.device_put(jnp.array([2.0, 0.75], F32)))
    assert (search.sync(), search.best, search.history_len) == (
        [True, False],
        2.0,
        7,
    ) and math.isclose(search.length, 1.6, rel_tol=1.0e-12)
    return prop.arms


def check_search(base, dimensions, leaves):
    blocks = [ParamBlock(*leaf, w) for leaf, w in zip(leaves, (1.0, 0.5, 2.0))]
    search = turbo_enn(base, -0.75, blocks, 8, max_pending=2, base_variance=0.04)
    search.profile(True)
    prop, bits = search.ask(2, 4, 1, 3, beta=1.3, draw_seed=23), _uints(base).tolist()
    batch = jax.dlpack.from_dlpack(prop)
    batch.block_until_ready()
    assert (
        prop.arms == 2
        and search.last_profile is not None
        and search.last_profile[3] > 0.0
        and batch.shape == (prop.arms, dimensions)
        and bool(jnp.all(jnp.isfinite(batch)).block_until_ready())
    )
    check_descriptions(prop, batch, bits, leaves, 0.8)
    _fails(partial(search.tell, prop, [1.0, 0.5], [0.01, 0.09]), "live JAX proposals")
    batch.delete()
    del batch
    gc.collect()
    search.tell(prop, [1.0, 0.5], [0.01, 0.09])
    assert (search.sync(), search.best, search.history_len, search.len) == (
        [True, False],
        1.0,
        3,
        dimensions,
    ) and math.isclose(search.best_variance, 0.01, rel_tol=1.0e-6)
    return _searchnext(search)


def _knndist(data, queries, values, neighbors: int, skip: int = 0):
    dists = np.sum(
        (queries.astype(F32)[:, None, :] - data.astype(F32)[None, :, :]) ** 2, axis=2
    )
    idx = np.argsort(dists, axis=1, kind="stable")[:, skip : skip + neighbors]
    return (
        idx,
        np.take_along_axis(dists, idx, axis=1),
        np.std(values, axis=0, ddof=0).astype(F32),
    )


def knn_expected(data, queries, values, skip=0, aleatoric=0.1):
    idx, nearest, scale = _knndist(data, queries, values, 6, skip=skip)
    weights = F32(1.0) / (F32(1.0e-9) + nearest + F32(aleatoric))
    norm = np.sum(weights, axis=1, dtype=F32)
    means = (
        np.sum(weights[:, :, None] * values.astype(F32)[idx], axis=1, dtype=F32)
        / norm[:, None]
    )
    return (
        idx,
        means,
        np.sqrt(np.maximum(F32(1.0) / norm, F32(1.0e-9)))[:, None] * scale[None, :],
    )


def knn_weighted(
    data, queries, values, variances, *, neighbors=6, epistemic=0.7, aleatoric=0.13
):
    idx, nearest, scale = _knndist(data, queries, values, neighbors)
    s_yvar = variances.astype(F32)[idx] / (scale[None, None, :] ** 2)
    weights = F32(1.0) / (
        F32(1.0e-9) + F32(epistemic) * nearest[:, :, None] + F32(aleatoric) + s_yvar
    )
    norm, s_row = np.sum(weights, axis=1, dtype=F32), scale[None, :]
    normed, epi_v = weights / norm[:, None, :], F32(1.0) / norm
    ale_v = np.sum(normed * (F32(aleatoric) + s_yvar), axis=1, dtype=F32)
    means = np.sum(normed * values.astype(F32)[idx], axis=1, dtype=F32)
    l2 = np.sqrt(np.sum(normed * normed, axis=1, dtype=F32))
    return (
        idx,
        means,
        np.sqrt(epi_v + ale_v) * s_row,
        np.sqrt(epi_v) * s_row,
        np.sqrt(ale_v) * s_row,
        normed,
        l2,
    )[:]


def _checkwpost(actual, expected) -> None:
    assert (
        arr_eq(actual[4], expected[0])
        and allclose(actual[:4], expected[1:5], rtol=5e-5, atol=5e-6) is None
    )


def _checkdraws(draws, draw_indices, expected, seeds) -> None:
    exp_d = draw_expected(
        expected[0], expected[5], expected[6], expected[1], expected[2], seeds
    )
    assert (
        draw_indices == expected[0].tolist()
        and allclose(draws, exp_d, rtol=2e-4, atol=2e-5) is None
    )


def _condcase(cuda, train, queries, values, variances, seeds) -> None:
    wx, wy = queries[:2] + 0.003, np.array([[1.25, -0.4], [0.8, 0.3]], dtype=F64)
    c_x, c_y = cat((train, wx), axis=0), cat((values, wy), axis=0)
    exp_c = knn_weighted(c_x, queries, c_y, cat((variances, np.zeros_like(wy)), axis=0))
    _checkwpost(
        cuda.conditional_posterior(wx, wy, queries, 6, 0.7, 0.13, False, True), exp_c
    )
    _checkdraws(
        *cuda.conditional_draw(wx, wy, queries, 6, 0.7, 0.13, seeds, False, True),
        exp_c,
        seeds,
    )


def weighted_case(train, queries, values) -> None:
    r, m = (
        np.arange(train.shape[0], dtype=F64)[:, None],
        np.arange(values.shape[1], dtype=F64)[None, :],
    )
    v = 0.02 + 0.01 * np.abs(np.sin(r * 0.071 + m * 0.29))
    cuda, exp = (
        ENN(train, values, v, False, "cuda"),
        knn_weighted(train, queries, values, v),
    )
    cuda.set_neighbors(False)
    _checkwpost(cuda.posterior(queries, 6, 0.7, 0.13, False, True), exp)
    refs, seeds = (
        [
            knn_weighted(
                train, queries, values, v, neighbors=k, epistemic=e, aleatoric=a
            )
            for k, e, a in zip((3, 6), (0.4, 0.7), (0.05, 0.13))
        ],
        [7, -3, 101],
    )
    for idx, obs in enumerate(
        cuda.batch_posterior(queries, [3, 6], [0.4, 0.7], [0.05, 0.13], False, True)
    ):
        allclose(obs, np.stack([ref[idx + 1] for ref in refs]), rtol=5e-5, atol=5e-6)
    _checkdraws(
        *cuda.posterior_draw(queries, 6, 0.7, 0.13, seeds, False, True), exp, seeds
    )
    _condcase(cuda, train, queries, values, v, seeds)
    restored = cuda.posterior(queries, 6, 0.7, 0.13, False, True)
    assert (
        arr_eq(restored[4], exp[0])
        and allclose(restored[0], exp[1], rtol=5e-5, atol=5e-6) is None
    )


def _checkkpost(cuda, train, queries, values, skip: int = 0) -> None:
    out, (idx, means, errs) = (
        cuda.posterior(queries, 6, 1.0, 0.1, skip > 0, False),
        knn_expected(train, queries, values, skip=skip),
    )
    assert (
        arr_eq(out[4], idx)
        and allclose(out[:2], (means, errs), rtol=3e-5, atol=3e-6) is None
    )


def knn_case(rows: int, dims: int) -> None:
    r, d = np.arange(rows, dtype=F64)[:, None], np.arange(dims, dtype=F64)[None, :]
    train, queries = (
        np.sin(r * 0.173 + d * 0.131),
        np.cos(np.arange(5, dtype=F64)[:, None] * 0.271 + d * 0.097),
    )
    values, cuda = (
        (v := cat((r / rows, np.sin(r * 0.113)), axis=1)),
        ENN(train, v, None, False, "cuda"),
    )
    cuda.set_neighbors(False)
    _checkkpost(cuda, train, queries, values)
    weighted_case(train, queries, values)
    extra, extra_v = (
        np.sin((r[:1] + rows) * 0.173 + d * 0.131),
        np.array([[1.5, -0.25]], dtype=F64),
    )
    cuda.add(extra, extra_v)
    train, values = cat((train, extra), axis=0), cat((values, extra_v), axis=0)
    _checkkpost(cuda, train, queries, values)
    _checkkpost(cuda, train, train[:3], values, skip=1)


def check_knn() -> None:
    for rows, dims in ((517, 7), (79, 73)):
        knn_case(rows, dims)


def _checkbuf(base, base_bits, leaves) -> None:
    terms, buf = (
        [(41, 0.25), (73, -0.125), (89, 0.03125)],
        ParameterBuffer(base, [ParamBlock(*lf) for lf in leaves]),
    )
    buf.materialize(terms)
    cand = jax.dlpack.from_dlpack(buf)
    assert _uints(cand).tolist() == cpu_values(base_bits, leaves, terms)
    _fails(partial(buf.materialize, [(97, 1.0)]), "candidate is alive")
    del cand
    gc.collect()
    buf.materialize([(97, 1.0)])
    bf = lambda x, s: ParameterBuffer(  # noqa: E731
        jax.device_put(jnp.array([x], jnp.uint16)).view(jnp.bfloat16),
        [ParamBlock(29, 0, 1, s)],
    )
    _fails(partial(bf, 0x7F80, 1.0), "finite")
    _fails(partial(bf(0x7F7F, MAX_F32).materialize, [(101, MAX_F32)]), "overflowed")


def main() -> None:
    leaves = [(17, 0, 257, 1.0e-6), (19, 257, 516, 1.0e-3), (23, 773, 257, 0.25)]
    base = jax.device_put(
        jnp.linspace(-4.0, 4.0, 1030, dtype=jnp.bfloat16).at[0].set(0.0).at[1].set(-0.0)
    )
    _checkbuf(base, (bits := _uints(base).tolist()), leaves)
    proposals = check_search(base, len(bits), leaves)
    check_singlesearch()
    check_export()
    check_knn()
    print(
        f"BF16_PARITY ok=true exact={len(bits)} proposals={proposals} "
        "leases=true noise=true profile=true validation=true knn=true weighted=true "
        "batch=true draws=true conditional=true device_tell=true "
        "nearest=true describe=true single_arm=true read_best=true failure_tolerance=true "
        "mutation_rejected=true"
    )


if __name__ == "__main__":
    main()
