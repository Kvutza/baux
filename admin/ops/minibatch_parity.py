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
    old_radius = search.length
    search, proposals = ask(search, 42 + step)
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
    search, _ = search.tell(proposals, value, variance)
    fails(search.incumbent, "sync")
    fails(partial(search.tell, proposals, value, variance))
    search, flags = search.sync()
    search, empty_flags = search.sync()
    assert flags == [accept] and empty_flags == []
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
    return search


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
            search = _updatestep(search, state, capacity, step, round_cfg)
        search, proposals = ask(search, 99)
        actual = read_proposals(proposals)[0]
        np.testing.assert_allclose(
            proposals.describe()[0][1],
            score(actual, state["hist"]),
            rtol=4e-5,
            atol=4e-5,
        )
        search, _ = search.tell(proposals, 0.0, 0.0)
        search, flags = search.sync()
        assert flags == [False]


def _checkbadpaired(search, proposals, valid):
    for index in range(len(valid)):
        invalids = [float("nan"), float("inf"), -float("inf"), 1e100]
        if index == 1:
            invalids.extend([-0.1, -1e-100])
        for value in invalids:
            args = valid.copy()
            args[index] = value
            fails(
                partial(search.tell, proposals, *args),
                error_types=(ValueError, OverflowError),
            )


def _checkother(other, foreign, valid):
    import jax.numpy as jnp

    other, _ = other.tell(foreign, *valid)
    other, flags = other.sync()
    assert flags == [True]
    other, proposals = ask(other, 43)
    fails(lambda: other.tell(proposals, [100.0]), "mix")
    fails(lambda: other.tell(proposals, jnp.asarray([100.0], dtype=jnp.float32)), "mix")
    fails(lambda: other.tell(foreign, *valid), "outstanding")
    other, _ = other.tell(proposals, *valid)
    other, flags = other.sync()
    assert flags == [True]
    for sampler in ("independent", "gaussian"):
        state, _, _ = fixture(sampler)
        state, pending = ask(state)
        fails(partial(state.tell, pending, *valid), "correlated")
        state, _ = state.tell(pending, [9.0])
        state, flags = state.sync()
        assert flags == [False]


def check_mixed():
    search, base, _ = fixture()
    search, proposals = ask(search)
    valid = [1.0, 0.0]
    _checkbadpaired(search, proposals, valid)
    other, _, _ = fixture()
    other, foreign = ask(other)
    fails(lambda: search.tell(foreign, *valid), "outstanding")
    fails(lambda: search.tell(foreign, [1.0]), "outstanding")
    search, _ = search.tell(proposals, [9.0], [0.5])
    search, flags = search.sync()
    assert flags == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    assert search.best == 10.0 and search.best_variance == 0.125
    search, proposals = ask(search, 43)
    fails(lambda: search.tell(proposals, *valid), "mix")
    search, _ = search.tell(proposals, [11.0])
    search, flags = search.sync()
    assert flags == [True]
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
    search, proposals = ask(search)
    fails(stale.__dlpack__, "stale")
    array = jax.dlpack.from_dlpack(proposals)
    for action in (
        search.incumbent,
        partial(ask, search),
        search.sync,
        partial(search.tell, proposals, [0.0]),
        partial(search.tell, proposals, 0.0, 0.0),
    ):
        fails(action, "release live")
    array.block_until_ready()
    array.delete()
    del array, stale, action
    gc.collect()
    search, _ = search.tell(proposals, 0.0, 0.0)
    search, flags = search.sync()
    assert flags == [False]
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
    search, pending = ask(search)
    replay, clean = ask(replay)
    assert complete.query(), "scratch reuse did not wait for the consumer stream"
    tensor = torch.from_dlpack(LegacyExport(pending))
    assert tensor.data_ptr() == scratch_pointer
    del tensor
    gc.collect()
    np.testing.assert_array_equal(read_proposals(pending), read_proposals(clean))
    search, _ = search.tell(pending, 20.0, 0.5)
    replay, _ = replay.tell(clean, 20.0, 0.5)
    search, search_flags = search.sync()
    replay, replay_flags = replay.sync()
    assert search_flags == replay_flags == [False]
    np.testing.assert_array_equal(search.read_best(), base)
    np.testing.assert_array_equal(read_proposals(search.incumbent()), base[None, :])


def check_scratch():
    import torch

    for delayed in (False, True):
        _scratchcase(delayed)
    for _accept in (False, True):
        search, _, _ = fixture()
        search, pending = ask(search)
        tensor = torch.from_dlpack(LegacyExport(pending))
        tensor.fill_(0.5)
        del tensor
        gc.collect()
        search, _ = search.tell(pending, 20.0, 0.5)
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
