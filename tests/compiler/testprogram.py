"""Execute the same multi-output state transition with independent compilers."""

import numpy as np
import pytest
from baux.experimental import TensorSpec, compile

SOURCE = """
module {
  func.func @main(%weight: tensor<4xf32>, %noise: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = stablehlo.add %weight, %noise : tensor<4xf32>
    %observed = stablehlo.multiply %next, %next : tensor<4xf32>
    return %next, %observed : tensor<4xf32>, tensor<4xf32>
  }
}
"""
INPUTS = (TensorSpec((4,)), TensorSpec((4,)))
OUTPUTS = {"parameters": TensorSpec((4,)), "residuals": TensorSpec((4,))}


@pytest.mark.parametrize(
    "backend,device",
    [("iree", "cpu"), ("iree", "metal"), ("xla", "cpu"), ("tvm", "cpu")],
)
def test_round(backend, device):
    executable = compile(SOURCE, INPUTS, OUTPUTS, backend=backend, device=device)
    weights = np.array([1, -2, 3, 0], dtype=np.float32)
    noise = np.array([0.25, -0.5, 0, 2], dtype=np.float32)
    resident = executable.put((weights, noise))
    reference = weights.copy()
    weights[:] = 99
    first = executable(resident)
    second = executable((first["parameters"], resident[1]))
    host = executable.get(second)
    np.testing.assert_array_equal(host["parameters"], reference + 2 * noise)
    np.testing.assert_array_equal(host["residuals"], (reference + 2 * noise) ** 2)
    np.testing.assert_array_equal(
        executable.get(first)["parameters"], reference + noise
    )
    with pytest.raises(TypeError, match="get"):
        np.asarray(first["parameters"])
    with pytest.raises(ValueError, match="signature"):
        executable({"wrong": resident})


def test_contract():
    with pytest.raises(ValueError, match="signature"):
        compile(SOURCE, INPUTS, (TensorSpec((3,)), TensorSpec((4,))))
    executable = compile(SOURCE, INPUTS, OUTPUTS)
    with pytest.raises(ValueError, match="expected"):
        executable.put((np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float32)))
    with pytest.raises(ValueError, match="belong"):
        executable((np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)))
    other = compile(SOURCE, INPUTS, OUTPUTS)
    foreign = other.put((np.zeros(4, np.float32), np.ones(4, np.float32)))
    with pytest.raises(ValueError, match="belong"):
        executable(foreign)


@pytest.mark.parametrize(
    "backend,device",
    [("iree", "cpu"), ("iree", "metal"), ("xla", "cpu"), ("tvm", "cpu")],
)
def test_forward(backend, device):
    import jax
    import jax.numpy as jnp
    import optree
    from baux._embeddings import mlp_forward

    def step(state):
        weights, noise, observation, target = state
        moved = jax.tree.map(lambda w, n: w + n, weights, noise)
        prediction = mlp_forward(moved, observation, act=jnp.tanh)
        residual = prediction - target
        return moved, residual, jnp.mean(residual * residual)

    rng = np.random.default_rng(64)
    weights = [
        {
            "weight": rng.normal(size=shape).astype(np.float32),
            "bias": np.zeros(shape[0], dtype=np.float32),
        }
        for shape in [(16, 32), (8, 16)]
    ]
    noise = optree.tree_map(
        lambda w: rng.normal(0, 0.01, size=w.shape).astype(np.float32), weights
    )
    observation = rng.normal(size=32).astype(np.float32)
    target = rng.normal(size=8).astype(np.float32)
    args = (weights, noise, observation, target)
    moved = optree.tree_map(lambda w, n: w + n, weights, noise)
    residual = mlp_forward(moved, observation, act=np.tanh) - target
    expected = (moved, residual, np.mean(residual * residual))

    def specs(tree):
        return optree.tree_map(lambda x: TensorSpec(x.shape, str(x.dtype)), tree)

    executable = compile(step, specs(args), backend=backend, device=device)
    actual = executable.get(executable(executable.put(args)))
    for result, reference in zip(
        optree.tree_leaves(actual), optree.tree_leaves(expected), strict=True
    ):
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5)


def test_dataclass():
    from dataclasses import dataclass, field

    from baux.experimental.program import register

    @register
    @dataclass(frozen=True)
    class State:
        parameters: object
        noise: object
        tag: str = field(default="round", metadata={"pytree_node": False})

    specs = State(*INPUTS)

    def step(state):
        moved = state.parameters + state.noise
        return {"parameters": moved, "residuals": moved * moved}

    executable = compile(step, specs)
    host = State(np.zeros(4, np.float32), np.ones(4, np.float32))
    resident = executable.put(host)
    assert isinstance(resident, State)
    assert resident.tag == "round"
    np.testing.assert_array_equal(
        executable.get(executable(resident))["parameters"], np.ones(4)
    )
    with pytest.raises(ValueError, match="signature"):
        executable(State(resident.parameters, resident.noise, tag="changed"))


@pytest.mark.parametrize(
    "backend,device",
    [("iree", "cpu"), ("iree", "metal"), ("xla", "cpu"), ("tvm", "cpu")],
)
def test_axes(backend, device):
    from jax import lax

    def step(state):
        left, right, bias = state
        product = lax.dot_general(left, right, (((2,), (0,)), ((1,), (1,))))
        expanded = lax.broadcast_in_dim(bias, (2, 3, 5), (1,))
        return {"prediction": product + expanded + np.float32(0.25), "state": bias}

    rng = np.random.default_rng(91)
    args = tuple(
        rng.normal(size=shape).astype(np.float32)
        for shape in [(3, 2, 4), (4, 2, 5), (3,)]
    )
    specs = tuple(TensorSpec(x.shape) for x in args)
    executable = compile(step, specs, backend=backend, device=device)
    actual = executable.get(executable(executable.put(args)))
    expected = (
        np.einsum("abk,kbc->bac", args[0], args[1]) + args[2][None, :, None] + 0.25
    )
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual["state"], args[2])


@pytest.mark.parametrize("backend", ["iree", "xla", "tvm"])
def test_scalar(backend):
    import jax.numpy as jnp

    def step(state):
        return {"counter": state["counter"], "value": jnp.asarray(0.25, jnp.float64)}

    specs = {"counter": TensorSpec((), "uint64"), "unused": TensorSpec((2,))}
    host = {"counter": np.array(2**40, np.uint64), "unused": np.ones(2, np.float32)}
    executable = compile(step, specs, backend=backend)
    actual = executable.get(executable(executable.put(host)))
    assert actual["counter"].dtype == np.dtype("uint64")
    assert actual["counter"].shape == ()
    assert actual["counter"] == 2**40
    assert actual["value"].dtype == np.dtype("float64")
    assert actual["value"] == 0.25


@pytest.mark.parametrize(
    "backend,device",
    [("iree", "cpu"), ("iree", "metal"), ("xla", "cpu"), ("tvm", "cpu")],
)
def test_pending(backend, device):
    import asyncio
    import gc

    executable = compile(SOURCE, INPUTS, OUTPUTS, backend=backend, device=device)
    resident = executable.put((np.ones(4, np.float32), np.ones(4, np.float32)))
    calls = [executable.submit(resident) for _ in range(8)]
    assert isinstance(calls[0].ready(), bool)
    del resident
    gc.collect()

    async def collect():
        return await asyncio.gather(*calls)

    for result in asyncio.run(collect()):
        np.testing.assert_array_equal(
            executable.get(result)["parameters"], np.full(4, 2)
        )
    with pytest.raises((ValueError, RuntimeError), match="consumed"):
        calls[0].wait()


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
)
def test_storage(dtype):
    import ml_dtypes

    kind = ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype
    host = np.array([0, 1], dtype=kind)
    if dtype == "uint64":
        host[1] = 2**63 + 7
    executable = compile(lambda value: value, TensorSpec((2,), dtype))
    result = executable.get(executable(executable.put(host)))
    assert result.dtype == host.dtype
    np.testing.assert_array_equal(result, host)


def test_residentmodel():
    from baux.experimental import ResidentModel

    model = ResidentModel(seed=42)
    assert model.parameters == 265_623_320
    assert model.rows == 8192
    tokens = np.zeros(model.rows, dtype=np.uint32)
    labels = np.zeros(model.rows, dtype=np.uint32)
    mask = np.ones(model.rows, dtype=np.uint8)
    call = model.submit(tokens, labels, mask)
    assert isinstance(call.ready(), bool)
    output = call.wait()
    host = model.get(output)
    assert host.token_losses.shape == (8192,)
    assert host.sequence_nlls.shape == (2,)
    assert np.isfinite(host.token_losses).all()
    assert np.isfinite(host.sequence_nlls).all()
    with pytest.raises(ValueError, match="consumed"):
        call.wait()
