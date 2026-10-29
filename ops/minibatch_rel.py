"""Relative paired BF16 parity helpers for minibatch search."""

import gc
from functools import partial

import numpy as np
from correlated_parity import decode, read_proposals

from baux.experimental import ParamBlock, SearchState


class LegacyExport:
    def __init__(self, source):
        self.source = source

    def __dlpack_device__(self):
        return self.source.__dlpack_device__()

    def __dlpack__(self, stream=None, **kwargs):
        return self.source.__dlpack__(stream=stream)


def fixture(sampler="correlated", capacity=2, value=10.0, variance=0.125):
    import jax.numpy as jnp

    bits = np.resize(np.array([0x3FC0, 0xBFC0], dtype=np.uint16), 513)
    leaves = [(17, 0, bits.size, 1.0)]
    search = SearchState(
        jnp.asarray(bits).view(jnp.bfloat16),
        value,
        [ParamBlock(*leaf) for leaf in leaves],
        capacity,
        base_variance=variance,
        sampler=sampler,
        reference_seed=123,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
    )
    return search, bits, leaves


def ask(search, seed=42):
    return search.ask(
        1, 4, min(2, search.history_len), seed, acquisition="ucb", beta=0.5
    )


def fails(action, text=None, error_types=(ValueError, BufferError)):
    try:
        action()
    except error_types as error:
        if text is not None:
            assert text.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError("invalid operation unexpectedly succeeded")


def score(bits, history, epistemic_scale=0.7, aleatoric_scale=0.05):
    row = decode(bits).astype(np.float64)
    distances = [
        np.square(row - decode(item[0]).astype(np.float64)).sum() for item in history
    ]
    weights = 1.0 / (
        1e-9
        + epistemic_scale * np.asarray(distances)
        + aleatoric_scale
        + [item[2] for item in history]
    )
    return np.dot(
        weights, [item[1] for item in history]
    ) / weights.sum() + 0.5 / np.sqrt(weights.sum())


def ask_relative(search, seed=42, acquisition="ucb"):
    return search.ask(
        1,
        4,
        search.history_len,
        seed,
        epistemic_scale=1.0,
        aleatoric_scale=0.0,
        y_scale=1.0,
        acquisition=acquisition,
        beta=0.5,
        draw_seed=seed + 1000,
    )


def _histstep(search, base, history, capacity, step, delta):
    proposals = ask_relative(search, 100 + step)
    actual = read_proposals(proposals)[0]
    np.testing.assert_allclose(
        proposals.describe()[0][1],
        score(actual, history, 1.0, 0.0),
        rtol=4e-5,
        atol=4e-5,
    )
    incumbent = 20.0 if step % 2 else -20.0
    paired_var = (step + 1) / 32
    accept = delta > 0
    search.tell_relative(
        proposals, incumbent + delta, 0.75, incumbent, 0.5, delta, paired_var, accept
    )
    assert search.sync() == [accept]
    if accept:
        base = actual.copy()
        history = [(base.copy(), 0.0, 0.0)]
    else:
        rejected = [*history[1:], (actual.copy(), delta, paired_var)]
        history = [history[0], *rejected[-(capacity - 1) :]]
    assert search.history_len == len(history) and search.restarts == 0
    assert search.best == (incumbent + delta if accept else incumbent)
    assert search.best_variance == (0.75 if accept else 0.5)
    np.testing.assert_array_equal(search.read_best(), base)
    return base, history


def check_history():
    deltas = [-0.25, -0.5, -0.125, -0.75, 0.125, -0.25, -0.5, 0.25, -0.125]
    for capacity in (2, 3):
        search, base, _ = fixture(capacity=capacity)
        search.enable_relative(failure_tolerance=4)
        assert search.best == 10.0 and search.best_variance == 0.125
        history = [(base.copy(), 0.0, 0.0)]
        for step, delta in enumerate(deltas):
            base, history = _histstep(search, base, history, capacity, step, delta)
        proposals = ask_relative(search, 200)
        actual = read_proposals(proposals)[0]
        np.testing.assert_allclose(
            proposals.describe()[0][1],
            score(actual, history, 1.0, 0.0),
            rtol=4e-5,
            atol=4e-5,
        )
        search.tell_relative(proposals, -21.0, 0.75, -20.0, 0.5, -1.0, 0.25, False)
        assert search.sync() == [False]


def _offsetstep(left, right, acquisition, step, delta):
    a = ask_relative(left, 300 + step, acquisition)
    b = ask_relative(right, 300 + step, acquisition)
    assert a.geometry() == b.geometry() and a.describe() == b.describe()
    np.testing.assert_array_equal(read_proposals(a), read_proposals(b))
    incumbent = -20.0 + step
    offset = 100.0 if step % 2 else -40.0
    accept = delta > 0
    left.tell_relative(
        a, incumbent + delta, 0.5, incumbent, 0.75, delta, 0.0625, accept
    )
    right.tell_relative(
        b,
        incumbent + offset + delta,
        1.5,
        incumbent + offset,
        1.75,
        delta,
        0.0625,
        accept,
    )
    assert left.sync() == right.sync() == [accept]
    assert right.best - left.best == offset
    assert right.best_variance - left.best_variance == 1.0
    assert left.history_len == right.history_len and left.length == right.length
    np.testing.assert_array_equal(left.read_best(), right.read_best())
    np.testing.assert_array_equal(left.read_reference(), right.read_reference())


def check_offsets():
    deltas = [-0.5, -0.25, -0.75, -0.125, 0.5, -0.25, -0.5]
    for acquisition in ("ucb", "thompson"):
        left, _, _ = fixture(capacity=3)
        right, _, _ = fixture(capacity=3, value=110.0, variance=0.875)
        for search in (left, right):
            search.enable_relative(failure_tolerance=4)
        for step, delta in enumerate(deltas):
            _offsetstep(left, right, acquisition, step, delta)


def check_contraction():
    search, base, _ = fixture()
    search.enable_relative(failure_tolerance=4)
    reference = search.read_reference()
    for step in range(12):
        proposals = ask_relative(search, 400 + step)
        search.tell_relative(proposals, 9.0, 0.5, 10.0, 0.75, -1.0, 0.25, False)
        assert search.sync() == [False] and search.restarts == 0
        assert search.length == max(0.03125, 0.125 / 2 ** ((step + 1) // 4))
        np.testing.assert_array_equal(search.read_best(), base)
        np.testing.assert_array_equal(search.read_reference(), reference)
    search, _, _ = fixture()
    search.enable_relative(failure_tolerance=4)
    failures = 0
    for step in range(8):
        old_radius = search.length
        proposals = ask_relative(search, 500 + step)
        radius = proposals.describe()[0][2]
        accept = step == 3
        delta = 1.0 if accept else -1.0
        search.tell_relative(
            proposals, 10.0 + delta, 0.5, 10.0, 0.75, delta, 0.25, accept
        )
        assert search.sync() == [accept]
        failures = 0 if accept else failures + 1
        expected = radius if accept else old_radius
        if failures == 4:
            expected, failures = max(0.03125, old_radius / 2), 0
        assert search.length == expected and search.restarts == 0


def _checkbadargs(search, proposals, valid):
    for index in range(6):
        invalids = [float("nan"), float("inf"), -float("inf"), 1e100]
        if index in (1, 3, 5):
            invalids += [-0.1, -1e-100]
        for value in invalids:
            args = valid.copy()
            args[index] = value
            fails(
                partial(search.tell_relative, proposals, *args),
                error_types=(ValueError, OverflowError),
            )


def _checkbadmodes(valid):
    for sampler in ("independent", "gaussian"):
        state, _, _ = fixture(sampler)
        fails(state.enable_relative)
        pending = ask(state)
        fails(partial(state.tell_relative, pending, *valid))
        state.tell(pending, [9.0])
        assert state.sync() == [False]
    for legacy in ("tell", "tell_paired"):
        state, _, _ = fixture()
        pending = ask(state)
        fails(state.enable_relative)
        fails(partial(state.tell_relative, pending, *valid))
        if legacy == "tell":
            state.tell(pending, [9.0])
        else:
            state.tell_paired(pending, 9.0, 0.5, 10.0, 0.75, False)
        assert state.sync() == [False]
        fails(state.enable_relative)


def check_validation():
    valid = [11.0, 0.5, 10.0, 0.75, 1.0, 0.25, True]
    undersized, _, _ = fixture(capacity=1)
    fails(undersized.enable_relative)
    search, _, _ = fixture()
    for tolerance in (0, -1, 2**32):
        fails(
            partial(search.enable_relative, failure_tolerance=tolerance),
            error_types=(ValueError, OverflowError),
        )
    search.enable_relative(failure_tolerance=4)
    fails(search.enable_relative)
    proposals = ask_relative(search)
    _checkbadargs(search, proposals, valid)
    fails(partial(search.tell, proposals, [100.0]))
    fails(partial(search.tell_paired, proposals, *valid[:4], True))
    foreign_search, _, _ = fixture()
    foreign_search.enable_relative()
    foreign = ask_relative(foreign_search)
    fails(partial(search.tell_relative, foreign, *valid))
    search.tell_relative(proposals, *valid)
    fails(partial(search.tell_relative, proposals, *valid))
    assert search.sync() == [True] and search.sync() == []
    fails(search.enable_relative)
    foreign_search.tell_relative(foreign, *valid)
    assert foreign_search.sync() == [True]
    _checkbadmodes(valid)


def check_pairedleases():
    import torch

    search, base, _ = fixture()
    snapshot = search.incumbent()
    tensor = torch.from_dlpack(LegacyExport(snapshot))
    fails(search.enable_relative, "release live")
    del tensor
    gc.collect()
    search.enable_relative()
    proposals = ask_relative(search)
    tensor = torch.from_dlpack(LegacyExport(proposals))
    tell = partial(
        search.tell_relative, proposals, 9.0, 0.5, 10.0, 0.75, -1.0, 0.25, False
    )
    fails(tell, "release live")
    fails(search.sync, "release live")
    del tensor
    gc.collect()
    tell()
    assert search.sync() == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    for accept in (False, True):
        state, _, _ = fixture()
        state.enable_relative()
        pending = ask_relative(state)
        tensor = torch.from_dlpack(LegacyExport(pending))
        tensor.fill_(0.5)
        del tensor
        gc.collect()
        state.tell_relative(pending, 11.0, 0.5, 10.0, 0.75, 1.0, 0.25, accept)
        fails(state.sync, "invalid")
