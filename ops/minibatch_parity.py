"""GPU parity for native paired BF16 acceptance and incumbent DLPack leases.

Run with the CUDA wheel: python ops/minibatch_parity.py
"""

import gc
from functools import partial

import numpy as np
from correlated_parity import candidate, check_rounding, read_proposals
from minibatch_rel import (
    LegacyExport,
    ask,
    ask_relative,
    check_contraction,
    check_history,
    check_offsets,
    check_pairedleases,
    check_validation,
    fails,
    fixture,
    score,
)

__all__ = [
    "LegacyExport",
    "ask",
    "ask_relative",
    "check_contract",
    "check_contraction",
    "check_history",
    "check_mixed",
    "check_offsets",
    "check_pairedleases",
    "check_scratch",
    "check_updates",
    "check_validation",
    "fails",
    "fixture",
    "main",
    "score",
]


def _updatestep(search, state, capacity, step, round_cfg):
    value, variance, incumbent, inc_var, accept = round_cfg
    snapshot = search.incumbent()
    np.testing.assert_array_equal(read_proposals(snapshot), state["base"][None, :])
    fails(snapshot.__dlpack__, "already exported")
    old_radius, proposals = search.length, ask(search, 42 + step)
    fails(search.incumbent, "outstanding")
    index, _ = proposals.geometry()[0]
    _, actual_score, radius, _ = proposals.describe()[0]
    actual = read_proposals(proposals)[0]
    np.testing.assert_allclose(
        actual_score, score(actual, state["hist"]), rtol=4e-5, atol=4e-5
    )
    expected = candidate(
        state["base"], state["leaves"], state["ref"], 42 + step, index, radius
    )
    check_rounding(actual, expected[3], expected[4])
    search.tell_paired(proposals, value, variance, incumbent, inc_var, accept)
    fails(search.incumbent, "sync")
    fails(
        partial(
            search.tell_paired, proposals, value, variance, incumbent, inc_var, accept
        )
    )
    assert search.sync() == [accept] and search.sync() == []
    assert search.best == (value if accept else incumbent)
    assert search.best_variance == (variance if accept else inc_var)
    assert search.restarts == 0 and search.length == (radius if accept else old_radius)
    stored = search.read_reference()
    if accept:
        check_rounding(stored, expected[1], expected[2])
        assert np.any(stored != state["ref"])
        state["base"] = actual
    else:
        np.testing.assert_array_equal(stored, state["ref"])
    state["ref"] = stored
    np.testing.assert_array_equal(search.read_best(), state["base"])
    state["hist"] = [*state["hist"], (actual.copy(), value, variance)][-capacity:]
    assert search.history_len == len(state["hist"])
    fails(proposals.__dlpack__, "outstanding")


def check_updates():
    rounds = [
        (20.0, 0.25, 21.0, 0.5, False),
        (-2.0, 0.75, -3.0, 0.125, True),
        (5.0, 0.5, -4.0, 0.25, False),
        (-8.0, 0.125, -7.0, 0.75, True),
    ]
    for capacity in (1, 2):
        search, base, leaves = fixture(capacity=capacity)
        state = {
            "base": base,
            "leaves": leaves,
            "ref": search.read_reference(),
            "hist": [(base.copy(), 10.0, 0.125)],
        }
        for step, round_cfg in enumerate(rounds):
            _updatestep(search, state, capacity, step, round_cfg)
        proposals = ask(search, 99)
        actual = read_proposals(proposals)[0]
        np.testing.assert_allclose(
            proposals.describe()[0][1],
            score(actual, state["hist"]),
            rtol=4e-5,
            atol=4e-5,
        )
        search.tell_paired(proposals, 0.0, 0.0, -8.0, 0.125, False)
        assert search.sync() == [False]


def _checkbadpaired(search, proposals, valid):
    for index in range(4):
        invalids = [float("nan"), float("inf"), -float("inf"), 1e100]
        if index in (1, 3):
            invalids.extend([-0.1, -1e-100])
        for value in invalids:
            args = valid.copy()
            args[index] = value
            fails(
                partial(search.tell_paired, proposals, *args),
                error_types=(ValueError, OverflowError),
            )


def _checkother(other, foreign, valid):
    import jax.numpy as jnp

    other.tell_paired(foreign, *valid)
    assert other.sync() == [True]
    proposals = ask(other, 43)
    fails(lambda: other.tell(proposals, [100.0]), "mix")
    fails(lambda: other.tell(proposals, jnp.asarray([100.0], dtype=jnp.float32)), "mix")
    fails(lambda: other.tell_paired(foreign, *valid), "outstanding")
    other.tell_paired(proposals, *valid)
    assert other.sync() == [True]
    for sampler in ("independent", "gaussian"):
        state, _, _ = fixture(sampler)
        pending = ask(state)
        fails(partial(state.tell_paired, pending, *valid), "correlated")
        state.tell(pending, [9.0])
        assert state.sync() == [False]


def check_mixed():
    search, base, _ = fixture()
    proposals = ask(search)
    valid = [1.0, 0.0, 0.0, 0.0, True]
    _checkbadpaired(search, proposals, valid)
    other, _, _ = fixture()
    foreign = ask(other)
    fails(lambda: search.tell_paired(foreign, *valid), "outstanding")
    fails(lambda: search.tell(foreign, [1.0]), "outstanding")
    search.tell(proposals, [9.0], [0.5])
    assert search.sync() == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    assert search.best == 10.0 and search.best_variance == 0.125
    proposals = ask(search, 43)
    fails(lambda: search.tell_paired(proposals, *valid), "mix")
    search.tell(proposals, [11.0])
    assert search.sync() == [True]
    _checkother(other, foreign, valid)


def _checkcapsules(search, base):
    import jax
    import jax.numpy as jnp

    for version in ((0, 8), (1, 0), (1, 3), (2, 0)):
        snapshot = search.incumbent()
        capsule = snapshot.__dlpack__(
            max_version=version, dl_device=(2, 0), copy=False, stream=-1
        )
        fails(search.incumbent, "release live")
        del capsule
        gc.collect()
        np.testing.assert_array_equal(search.read_best(), base)
    snapshot = search.incumbent()
    array = jax.dlpack.from_dlpack(snapshot)
    del snapshot, search
    gc.collect()
    np.testing.assert_array_equal(np.asarray(array.view(jnp.uint16)), base[None, :])
    array.delete()


def _checkincumbent(search, base):
    import jax
    import jax.numpy as jnp

    old, current = search.incumbent(), search.incumbent()
    fails(old.__dlpack__, "stale")
    for kwargs in (
        {"copy": True},
        {"dl_device": (2, 1)},
        {"dl_device": (1, 0)},
        {"stream": 0},
        {"stream": -2},
    ):
        fails(partial(current.__dlpack__, **kwargs))
    assert current.__dlpack_device__() == (2, 0)
    array = jax.dlpack.from_dlpack(current)
    assert array.shape == (1, base.size) and array.dtype == jnp.bfloat16
    assert next(iter(array.devices())).id == 0
    for action in (
        partial(ask, search),
        search.incumbent,
        search.sync,
        search.read_best,
        search.read_reference,
        partial(search.profile, True),
        partial(getattr, search, "length"),
        current.__dlpack__,
    ):
        fails(action, "release live")
    np.testing.assert_array_equal(np.asarray(array.view(jnp.uint16)), base[None, :])
    array.delete()
    del array
    gc.collect()
    fails(current.__dlpack__, "already exported")


def check_contract():
    import jax

    search, base, _ = fixture()
    _checkincumbent(search, base)
    stale = search.incumbent()
    proposals = ask(search)
    fails(stale.__dlpack__, "stale")
    array = jax.dlpack.from_dlpack(proposals)
    for action in (
        search.incumbent,
        partial(ask, search),
        search.sync,
        partial(search.tell, proposals, [0.0]),
        partial(search.tell_paired, proposals, 0.0, 0.0, 0.0, 0.0, False),
    ):
        fails(action, "release live")
    array.block_until_ready()
    array.delete()
    del array, stale, action
    gc.collect()
    search.tell_paired(proposals, 0.0, 0.0, 10.0, 0.125, False)
    assert search.sync() == [False]
    del proposals
    _checkcapsules(search, base)


def _fillscratch(snapshot, delayed):
    import torch

    stream, complete = torch.cuda.Stream(), torch.cuda.Event()
    with torch.cuda.stream(stream):
        tensor = torch.from_dlpack(LegacyExport(snapshot))
        scratch_pointer = tensor.data_ptr()
        if delayed:
            torch.cuda._sleep(200_000_000)
        tensor.fill_(float("nan"))
        complete.record()
    return tensor, scratch_pointer, complete


def _scratchcase(delayed):
    import torch

    search, base, _ = fixture()
    replay, _, _ = fixture()
    tensor, scratch_pointer, complete = _fillscratch(search.incumbent(), delayed)
    fails(search.incumbent, "release live")
    del tensor
    gc.collect()
    pending, clean = ask(search), ask(replay)
    assert complete.query(), "scratch reuse did not wait for the consumer stream"
    tensor = torch.from_dlpack(LegacyExport(pending))
    assert tensor.data_ptr() == scratch_pointer
    del tensor
    gc.collect()
    np.testing.assert_array_equal(read_proposals(pending), read_proposals(clean))
    search.tell_paired(pending, 20.0, 0.5, 21.0, 0.25, False)
    replay.tell_paired(clean, 20.0, 0.5, 21.0, 0.25, False)
    assert search.sync() == replay.sync() == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    np.testing.assert_array_equal(read_proposals(search.incumbent()), base[None, :])


def check_scratch():
    import torch

    for delayed in (False, True):
        _scratchcase(delayed)
    for accept in (False, True):
        search, _, _ = fixture()
        pending = ask(search)
        tensor = torch.from_dlpack(LegacyExport(pending))
        tensor.fill_(0.5)
        del tensor
        gc.collect()
        search.tell_paired(pending, 20.0, 0.5, 21.0, 0.25, accept)
        fails(search.sync, "invalid")


def main():
    import jax

    assert jax.devices()[0].platform == "gpu", "requires CUDA device 0"
    check_updates()
    check_mixed()
    check_contract()
    check_scratch()
    check_history()
    check_offsets()
    check_contraction()
    check_validation()
    check_pairedleases()
    print(
        "MINIBATCH_PARITY ok=true paired=true fifo=true variance=true reference=true "
        "radius=true mixed_api=true leases=true scratch_reused=true mutation=true "
        "paired_relative=true pinned_anchor=true recenter=true batch_offsets=true "
        "failure_contraction=true relative_validation=true relative_leases=true"
    )


if __name__ == "__main__":
    main()
