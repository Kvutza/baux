"""Validate dense Gaussian CUDA proposals; CPU libm agreement is approximate."""

import gc
import sys
from pathlib import Path

import click
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.correlated_cpu import (  # noqa: E402
    EPS,
    MASK64,
    NOISE_DOMAIN,
    REFERENCE_DOMAIN,
    candidate,
    candidate_seed,
    check_cpuhelpers,
    check_rounding,
    decode,
    dense_normal,
    direction,
    draw_normal,
    encode,
    f32,
    initial_reference,
    mix64,
    null_case,
    rounded_bounds,
    score,
    score_bounds,
    trial_hash,
)

__all__ = [
    "EPS",
    "MASK64",
    "NOISE_DOMAIN",
    "REFERENCE_DOMAIN",
    "append_history",
    "ask",
    "candidate",
    "candidate_seed",
    "check_changes",
    "check_copyorder",
    "check_correlated",
    "check_cpuhelpers",
    "check_gaussian",
    "check_nulls",
    "check_rounding",
    "check_tiles",
    "decode",
    "dense_normal",
    "direction",
    "draw_normal",
    "encode",
    "f32",
    "fixture",
    "initial_reference",
    "main",
    "mix64",
    "null_case",
    "read_proposals",
    "rounded_bounds",
    "score",
    "score_bounds",
    "trial_hash",
]


def read_proposals(proposals):
    import jax
    import jax.numpy as jnp

    batch = jax.dlpack.from_dlpack(proposals)
    try:
        return np.array(
            jax.device_get(jax.lax.bitcast_convert_type(batch, jnp.uint16)), copy=True
        )
    finally:
        batch.delete()
        del batch
        gc.collect()


def check_changes(changes, leaves, old, actual):
    assert len(changes) == len(leaves)
    for (_, off, ln, _), (changed, squared) in zip(leaves, changes):
        before, after = old[off : off + ln], actual[off : off + ln]
        assert changed == np.count_nonzero(before != after)
        delta = decode(after).astype(np.float64) - decode(before).astype(np.float64)
        np.testing.assert_allclose(
            squared, np.square(delta).sum(), rtol=3e-6, atol=1e-10
        )


def append_history(history, bits, reward):
    slot = len(history) + 1 if len(history) < 2 else history.pop(0)[0]
    history.append((slot, decode(bits).astype(np.float64), reward))


def fixture(sampler):
    import jax
    import jax.numpy as jnp

    from baux.experimental import ParamBlock, turbo_enn

    leaves = [
        (17, 0, 1, 2**-20),
        (19, 1, 30, 0.03125),
        (23, 31, 3, 0.5),
        (29, 34, 513, 1.0),
        (31, 547, 485, 2.0),
    ]
    size = sum(leaf[2] for leaf in leaves)
    bits = np.resize(np.asarray([0x3FC0, 0xBFC0], dtype=np.uint16), size)
    base = jax.device_put(jnp.asarray(bits)).view(jnp.bfloat16)
    search = turbo_enn(
        base,
        0.0,
        [ParamBlock(*leaf) for leaf in leaves],
        2,
        sampler=sampler,
        reference_seed=0x123456789ABCDEF0,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
        failure_tolerance=4 if sampler == "gaussian" else None,
    )
    return leaves, bits, search


def ask(search, root, draw):
    return search.ask(
        1,
        4,
        2,
        root,
        epistemic_scale=0.7,
        aleatoric_scale=0.05,
        y_scale=1.0,
        acquisition="thompson",
        draw_seed=draw,
    )


def _verifyask(prop, rep, exp, state, radii, draw):
    idx, desc = prop.geometry()[0][0], prop.describe()[0]
    state["seen"].add(idx)
    actual = read_proposals(prop)[0]
    assert 0 <= idx < 4 and prop.geometry()[0][1] == (0.75 if idx < 2 else 0.0)
    assert desc[0] == exp[idx][0] and desc[2] == radii[idx]
    assert np.array_equal(actual, read_proposals(rep)[0])
    np.testing.assert_allclose(
        desc[1], score(actual, state["hist"], draw), rtol=3e-5, atol=3e-5
    )
    bounds = [score_bounds(item[3], item[4], state["hist"], draw) for item in exp]
    slack = 6e-5 * max(1, abs(desc[1]))
    assert desc[1] >= max(b[0] for b in bounds) - slack
    assert bounds[idx][0] - slack <= desc[1] <= bounds[idx][1] + slack
    check_changes(desc[3], state["leaves"], state["base"], actual)
    return actual, idx, check_rounding(actual, exp[idx][3], exp[idx][4])


def _corrstep(search, replay, state, step):
    root, draw = mix64(step + 100), mix64(step + 900)
    radii = [
        f32(max(0.03125, state["radius"] / 2)),
        f32(min(0.5, state["radius"] * 2)),
    ] * 2
    exp = [
        candidate(state["base"], state["leaves"], state["ref"], root, i, radii[i])
        for i in range(4)
    ]
    for i in (0, 2):
        assert exp[i][0] == exp[i + 1][0] and np.array_equal(exp[i][1], exp[i + 1][1])
    prop, rep = ask(search, root, draw), ask(replay, root, draw)
    assert prop.geometry() == rep.geometry() and prop.describe() == rep.describe()
    actual, idx, mm = _verifyask(prop, rep, exp, state, radii, draw)
    state["mm"] += mm
    improve = step % 3 == 0
    reward = state["best"] + 1.0 if improve else state["best"] - 0.5
    for st, pend in ((search, prop), (replay, rep)):
        st.tell(pend, [reward], [0.0])
        assert st.sync() == [improve]
    append_history(state["hist"], actual, reward)
    stored = search.read_reference()
    if improve:
        state["mm"] += check_rounding(stored, exp[idx][1], exp[idx][2])
        state["best"], state["radius"], state["base"] = reward, radii[idx], actual
    else:
        np.testing.assert_array_equal(stored, state["ref"])
    state["ref"] = stored
    np.testing.assert_array_equal(state["ref"], replay.read_reference())
    for st in (search, replay):
        assert st.length == state["radius"] and st.best == state["best"]
        assert st.restarts == 0 and st.history_len == 2
        np.testing.assert_array_equal(st.read_best(), state["base"])


def check_correlated():
    leaves, base, search = fixture("correlated")
    _, _, replay = fixture("correlated")
    ref, init = search.read_reference(), initial_reference(leaves, 0x123456789ABCDEF0)
    mismatches = check_rounding(ref, init, 16 * EPS * (1 + abs(init)))
    np.testing.assert_array_equal(ref, replay.read_reference())
    state = {
        "leaves": leaves,
        "base": base,
        "ref": ref,
        "radius": 0.125,
        "best": 0.0,
        "mm": mismatches,
        "hist": [(1, decode(base).astype(np.float64), 0.0)],
        "seen": set(),
    }
    for step in range(48):
        _corrstep(search, replay, state, step)
    assert state["seen"] == {0, 1, 2, 3}, state["seen"]
    return state["mm"]


def _gaussstep(search, leaves, base, history, step):
    root, draw, radius = mix64(step + 70), mix64(step + 90), search.length
    prop = ask(search, root, draw)
    idx, desc = prop.geometry()[0][0], prop.describe()[0]
    assert 0 <= idx < 4 and prop.geometry()[0][1] == 0
    assert desc[0] == candidate_seed(root, idx) and desc[2] == f32(radius)
    exp = [
        candidate(base, leaves, None, root, i, radius, correlated=False)
        for i in range(4)
    ]
    assert len({item[0] for item in exp}) == 4
    actual = read_proposals(prop)[0]
    mm = check_rounding(actual, exp[idx][3], exp[idx][4])
    np.testing.assert_allclose(
        desc[1], score(actual, history, draw), rtol=3e-5, atol=3e-5
    )
    lower = [score_bounds(item[3], item[4], history, draw)[0] for item in exp]
    assert desc[1] >= max(lower) - 6e-5 * max(1, abs(desc[1]))
    check_changes(desc[3], leaves, base, actual)
    search.tell(prop, [-1.0])
    assert search.sync() == [False]
    append_history(history, actual, -1.0)
    np.testing.assert_array_equal(search.read_best(), base)
    return mm


def check_gaussian():
    leaves, base, search = fixture("gaussian")
    try:
        search.read_reference()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Independent Gaussian sampling must not expose a reference"
        )
    history = [(1, decode(base).astype(np.float64), 0.0)]
    mismatches = sum(
        _gaussstep(search, leaves, base, history, step) for step in range(4)
    )
    assert search.length == 0.0625 and search.restarts == 0
    return mismatches


def _checkrestart(search, expected):
    assert search.restarts == 1 and search.history_len == 1
    assert search.length == 0.125 and search.best == 4.0
    np.testing.assert_array_equal(search.read_best(), expected[3])
    prop = search.ask(1, 4, 1, 987, acquisition="thompson", draw_seed=654)
    actual = read_proposals(prop)[0]
    history = [(1, decode(expected[3]).astype(np.float64), 4.0)]
    np.testing.assert_allclose(
        prop.describe()[0][1], score(actual, history, 654), rtol=3e-5, atol=3e-5
    )
    search.tell(prop, [-1.0])
    assert search.sync() == [False]


def check_copyorder():
    import jax.numpy as jnp

    from baux.experimental import ParamBlock, turbo_enn

    size = 262_147
    search = turbo_enn(
        jnp.ones(size, dtype=jnp.bfloat16),
        0.0,
        [ParamBlock(17, 0, size, 1.0)],
        2,
        max_pending=5,
        sampler="gaussian",
        failure_tolerance=1,
        length_init=0.125,
        length_min=0.125,
        length_max=0.5,
    )
    prop = search.ask(5, 2, 1, 123)
    expected, rewards = read_proposals(prop), [1.0, 3.0, 2.0, 4.0, 0.0]
    search.tell(prop, rewards)
    assert search.sync() == [True, True, False, True, False]
    assert search.history_len == 2 and search.best == 4.0
    np.testing.assert_array_equal(search.read_best(), expected[3])
    history = [
        (1, decode(np.full(size, 0x3F80, dtype=np.uint16)).astype(np.float64), 0.0)
    ]
    for bits, reward in zip(expected, rewards):
        append_history(history, bits, reward)
    prop = ask(search, 456, 789)
    actual = read_proposals(prop)[0]
    np.testing.assert_allclose(
        prop.describe()[0][1], score(actual, history, 789), rtol=3e-5, atol=3e-5
    )
    search.tell(prop, [-1.0])
    assert search.sync() == [False]
    _checkrestart(search, expected)


def check_nulls():
    import jax.numpy as jnp

    from baux.experimental import ParamBlock, turbo_enn

    leaves, bits, radius, root = null_case()
    base = jnp.asarray(bits).view(jnp.bfloat16)
    for sampler, step in (
        ("correlated", radius),
        ("gaussian", radius),
        ("correlated", 1e-7),
    ):
        all_null = step == 1e-7
        reward = 2.0 if all_null else 0.0
        search = turbo_enn(
            base,
            reward,
            [ParamBlock(*leaf) for leaf in leaves],
            2,
            sampler=sampler,
            reference_seed=123,
            length_init=step,
            length_min=step / 4,
            length_max=step * 4,
        )
        prop = search.ask(1, 4, 1, root, acquisition="ucb", beta=0)
        idx, desc = prop.geometry()[0][0], prop.describe()[0]
        actual = read_proposals(prop)[0]
        assert desc[1] == reward
        check_changes(desc[3], leaves, bits, actual)
        if sampler == "correlated" and not all_null:
            assert idx == 1 and sum(count for count, _ in desc[3]) > 0
        else:
            assert idx == 0 and desc[3] == [(0, 0.0)]
            np.testing.assert_array_equal(actual, bits)
        search.tell(prop, [reward])
        assert search.sync() == [False]


def check_tiles():
    import jax.numpy as jnp

    from baux.experimental import ParamBlock, turbo_enn

    leaves, base = [(17, 0, 65_539, 1.0)], encode(np.full(65_539, 1.5))
    search = turbo_enn(
        jnp.asarray(base).view(jnp.bfloat16),
        0.0,
        [ParamBlock(*leaves[0])],
        2,
        sampler="correlated",
        reference_seed=123,
        length_init=0.125,
        length_min=0.03125,
        length_max=0.5,
    )
    ref, init = search.read_reference(), initial_reference(leaves, 123)
    mismatches = check_rounding(ref, init, 16 * EPS * (1 + abs(init)))
    for step in range(2):
        prop = search.ask(1, 4, 2, step, acquisition="ucb", beta=0, epistemic_scale=0)
        geom, desc = prop.geometry()[0], prop.describe()[0]
        exp = candidate(base, leaves, ref, step, geom[0], desc[2])
        assert geom == (0, 0.75) and desc[0] == exp[0]
        actual = read_proposals(prop)[0]
        mismatches += check_rounding(actual, exp[3], exp[4])
        check_changes(desc[3], leaves, base, actual)
        search.tell(prop, [1.0 if step == 0 else -1.0])
        assert search.sync() == [step == 0]
        stored = search.read_reference()
        if step == 0:
            mismatches += check_rounding(stored, exp[1], exp[2])
            base = actual
        else:
            np.testing.assert_array_equal(stored, ref)
        ref = stored
        np.testing.assert_array_equal(search.read_best(), base)
    return mismatches


def main(cpu_only):
    check_cpuhelpers()
    if cpu_only:
        click.echo(
            "CORRELATED_CPU_HELPERS ok=true gaussian=true rounding_boundaries=true"
        )
        return
    import jax

    from ops.bf16_parity import check_export

    device = jax.devices()[0]
    if device.platform != "gpu" or "T4" not in device.device_kind:
        raise click.ClickException(f"Expected a CUDA T4, got {device}")
    mismatches = check_correlated() + check_gaussian() + check_tiles()
    check_copyorder()
    check_nulls()
    check_export("correlated")
    click.echo(
        "CORRELATED_PARITY ok=true cpu=rounding_boundary_tolerance gpu_replay=exact "
        "acquisition=true reference=true radius=true gaussian=true fifo_copy=true "
        "reference_tiles=true restart_copy=true null_filter=true "
        f"cpu_boundary_mismatches={mismatches}"
    )


for _dec in (
    click.option(
        "--cpu-only",
        is_flag=True,
        help="Run numerical helper checks without CUDA imports.",
    ),
    click.command(help=__doc__),
):
    main = _dec(main)


if __name__ == "__main__":
    main()
