"""Compiler-visible production FBT and independent selection/loss checks."""

import numpy as np
import pytest
from baux.experimental import Fbt, ResidentModel, TensorSpec, compile


def specs(values):
    return tuple(TensorSpec(value.shape, str(value.dtype)) for value in values)


def test_export():
    model = Fbt()
    source, outputs = model.lower()
    assert model.inputs[0].shape == (265623320,)
    assert model.visits == (0, 1, 2, 1, 2, 3, 4)
    assert outputs == (TensorSpec((2, 4096)), TensorSpec((2,)))
    assert "stablehlo.custom_call" not in source
    assert "stablehlo.dot_general" in source
    assert "stablehlo.gather" in source


@pytest.mark.parametrize("backend", ["xla", "iree", "tvm"])
def test_tail(backend):
    from baux.experimental.fbt import readout

    # The peak is so dominant that log(sum(exp)) would round the loss to zero.
    hidden = np.ones((1, 1, 1), np.float16)
    embedding = np.full((1, 8192), -30, np.float16)
    embedding[0, 4100] = 0
    labels = np.array([[4100]], np.uint32)
    values = (hidden, embedding, labels)
    executable = compile(lambda x: readout(*x), specs(values), backend=backend)
    result = executable.get(executable(executable.put(values)))
    expected = np.log1p(8191 * np.exp(-30))
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-12)


def expected(query, key, position):
    summaries = [
        key.astype(np.float32).reshape((-1, 64, 64)).mean(axis=1).astype(np.float16)
    ]
    while len(summaries[-1]) > 1:
        values = summaries[-1].astype(np.float32)
        summaries.append(((values[::2] + values[1::2]) * 0.5).astype(np.float16))
    depth = max(0, (len(key) // 64 - 1).bit_length() - 4)
    candidates = list(range((len(key) // 64) >> depth))
    current = position // 64
    for level in range(depth, -1, -1):
        scored = []
        for node in candidates:
            begin, end = node << level, ((node + 1) << level) - 1
            if any(
                begin <= forced <= end for forced in (0, max(0, current - 1), current)
            ):
                score = np.inf
            elif end < current:
                vectors = (
                    summaries[level - 1][2 * node : 2 * node + 2]
                    if level
                    else key[node * 64 : (node + 1) * 64]
                )
                logits = query.astype(np.float32) @ vectors.astype(np.float32).T * 0.125
                maximum = logits.max(axis=-1, keepdims=True)
                score = (
                    maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=-1))
                ).sum()
            else:
                continue
            scored.append((node, score))
        chosen = [
            node for node, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:8]
        ]
        candidates = [child for node in chosen for child in (2 * node, 2 * node + 1)]
    return chosen + [-1] * (8 - len(chosen))


@pytest.mark.parametrize("backend", ["xla", "iree", "tvm"])
def test_refinement(backend):
    from baux.experimental._pisa import blocks

    rng = np.random.default_rng(94)
    positions = np.array([0, 63, 64, 511, 2048, 4095], np.int32)
    query = rng.normal(size=(1, len(positions), 8, 64)).astype(np.float16)
    key = rng.normal(size=(1, 4096, 64)).astype(np.float16)
    values = (query, key, positions)
    executable = compile(lambda x: blocks(*x), specs(values), backend=backend)
    actual = executable.get(executable(executable.put(values)))
    reference = np.array(
        [expected(q, key[0], int(p)) for q, p in zip(query[0], positions, strict=True)]
    )
    np.testing.assert_array_equal(actual[0], reference)


@pytest.mark.parametrize(
    ("backend", "device"),
    [("xla", "cpu"), ("iree", "cpu"), ("iree", "metal"), ("tvm", "cpu")],
)
def test_routing(backend, device):
    from baux.experimental import _routing
    from baux.experimental._hyper import dot

    native = ResidentModel(seed=19)
    rng = np.random.default_rng(84)
    tokens = rng.integers(0, 262144, native.rows, dtype=np.uint32)
    snapshots = dict(native._native.inspect(tokens, 128))
    packed = native.weights()
    graph = Fbt(context=128, batch=1)
    weights = {
        name: packed[offset : offset + int(np.prod(shape))].reshape(shape)
        for name, offset, shape in graph.layout
    }

    def branch(arguments):
        import jax.numpy as jnp

        value, router, gate, down = arguments
        logits = dot(value, router).astype(jnp.float16)
        ids = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), logits.shape)
        selected, _ = _routing.select(logits.astype(jnp.float32), ids, 3)
        return (
            _routing.forward(value, router, gate, down),
            logits,
            selected.astype(jnp.uint32),
        )

    executable = None
    for step, layer in enumerate(graph.visits):
        value = (
            np.frombuffer(snapshots[f"{step}/input"], np.float16)
            .reshape(128, 512)
            .copy()
        )
        arguments = (
            value,
            weights["router"][layer],
            weights["expert_gate_up"][layer],
            weights["expert_down"][layer],
        )
        if executable is None:
            executable = compile(
                branch, specs(arguments), backend=backend, device=device
            )
        output, scores, selected = executable.get(executable(executable.put(arguments)))
        expected = np.frombuffer(snapshots[f"{step}/branch"], np.float16).reshape(
            128, 512
        )
        expert_ids = np.frombuffer(snapshots[f"{step}/experts"], np.uint32).reshape(
            128, 3
        )
        native_scores = np.frombuffer(snapshots[f"{step}/scores"], np.float16).reshape(
            128, 64
        )
        print(
            f"same-input {backend}/{device} visit={step}: branch={np.max(np.abs(output.astype(np.float32) - expected))} score={np.max(np.abs(scores.astype(np.float32) - native_scores))} routes={np.count_nonzero(selected != expert_ids)}",
            flush=True,
        )
        np.testing.assert_array_equal(selected, expert_ids)
        np.testing.assert_allclose(output, expected, atol=0.004, rtol=0)
