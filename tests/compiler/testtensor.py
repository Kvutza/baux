"""Native compiler coverage for tensor indexing and a causal decoder model."""

import numpy as np
import pytest
from baux.experimental import TensorSpec, compile


def specification(tree):
    import jax

    return jax.tree.map(lambda x: TensorSpec(x.shape, str(x.dtype)), tree)


def check(function, arguments, expected):
    import jax

    executable = compile(function, specification(arguments), backend="tvm")
    pending = executable.submit(executable.put(arguments))
    actual = executable.get(pending.wait())
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert value.dtype == reference.dtype
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-5)
    return actual


def test_transforms():
    import jax.numpy as jnp
    from jax import lax

    def model(x):
        sliced = x[1:4:2, ::2]
        reverse = jnp.flip(sliced, axis=(0, 1))
        padded = lax.pad(reverse, jnp.float32(-2), ((1, 0, 1), (0, 2, 0)))
        indices = jnp.arange(4, dtype=jnp.int32)
        selected = jnp.where(indices < 2, indices, -indices)
        return (
            padded,
            jnp.concatenate((x, x), axis=1),
            selected,
            selected.astype(jnp.float32),
        )

    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    reverse = x[1:4:2, ::2][::-1, ::-1]
    padded = np.full((4, 5), -2, np.float32)
    padded[1::2, :3] = reverse
    selected = np.array([0, 1, -2, -3], np.int32)
    check(
        model,
        x,
        (padded, np.concatenate((x, x), axis=1), selected, selected.astype(np.float32)),
    )


def test_rounding():
    from jax import lax

    x = np.array(
        [
            -np.inf,
            -8388609,
            -3.5,
            -2.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            2.5,
            3.5,
            8388609,
            np.inf,
            np.nan,
        ],
        np.float32,
    )
    away = np.array(
        [-np.inf, -8388609, -4, -3, -1, -0.0, 0.0, 1, 3, 4, 8388609, np.inf, np.nan],
        np.float32,
    )

    def model(x):
        return (
            lax.floor(x),
            lax.ceil(x),
            lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN),
            lax.round(x, lax.RoundingMethod.AWAY_FROM_ZERO),
            lax.is_finite(x),
        )

    actual = check(
        model,
        x,
        (
            np.floor(x),
            np.ceil(x),
            np.rint(x),
            away,
            np.isfinite(x),
        ),
    )
    np.testing.assert_array_equal(actual[3], away)
    np.testing.assert_array_equal(np.signbit(actual[3][5:7]), [True, False])


def test_sign():
    from jax import lax

    floating = np.array([-np.inf, -3.25, -0.0, 0.0, 3.25, np.inf, np.nan], np.float32)
    integer = np.array([-2147483648, -5, -1, 0, 1, 5, 2147483647], np.int32)

    def model(values):
        x, n = values
        return (
            lax.sign(x),
            lax.rem(x, np.float32(2)),
            lax.sign(n),
            lax.rem(n, np.int32(2)),
        )

    with np.errstate(invalid="ignore"):
        remainder = np.fmod(floating, 2)
    # StableHLO sign preserves signed zero and propagates NaNs.
    expected = np.where(floating == 0, floating, np.sign(floating))
    actual = check(
        model,
        (floating, integer),
        (expected, remainder, np.sign(integer), np.fmod(integer, 2)),
    )
    np.testing.assert_array_equal(np.signbit(actual[0][2:4]), [True, False])


@pytest.mark.parametrize("start", [-5, 1, 100])
def test_dynamic(start):
    from jax import lax

    def model(state):
        x, patch, position = state
        return (
            lax.dynamic_slice(x, (position, np.int32(1)), (2, 2)),
            lax.dynamic_update_slice(x, patch, (position, np.int32(1))),
        )

    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    patch = np.full((2, 2), -7, np.float32)
    # lax wraps negative indices once before StableHLO clamps the start.
    clamped = max(0, min(start + 4 if start < 0 else start, 2))
    updated = x.copy()
    updated[clamped : clamped + 2, 1:3] = patch
    check(
        model,
        (x, patch, np.array(start, np.int32)),
        (x[clamped : clamped + 2, 1:3], updated),
    )


def test_gather():
    from jax import lax

    def model(state):
        x, indices = state
        dimensions = lax.GatherDimensionNumbers(
            offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=(0, 1)
        )
        return lax.gather(x, indices, dimensions, slice_sizes=(2, 3), mode="clip")

    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    indices = np.array([[-2, 1], [3, 99], [2, 0]], np.int32)
    expected = np.stack(
        [
            x[
                max(0, min(r, 3)) : max(0, min(r, 3)) + 2,
                max(0, min(c, 3)) : max(0, min(c, 3)) + 3,
            ]
            for r, c in indices
        ]
    )
    check(model, (x, indices), expected)


def test_batching():
    from jax import lax

    def model(state):
        x, indices = state
        dimensions = lax.GatherDimensionNumbers(
            offset_dims=(2,),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        )
        return lax.gather(x, indices, dimensions, (1, 1, 3), mode="clip")

    x = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    indices = np.array([[[3], [-1]], [[0], [99]]], np.int32)
    expected = np.stack([x[0, [3, 0]], x[1, [0, 3]]])
    check(model, (x, indices), expected)


@pytest.mark.parametrize("implicit", [False, True])
def test_indices(implicit):
    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    if implicit:
        indices = np.array([3, -2, 99], np.int32)
        dimensions = "offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1"
        sizes, output = "1, 6", (3, 6)
        expected = x[[3, 0, 4]]
    else:
        indices = np.array([[1, -2, 99], [2, 0, 99]], np.int32)
        dimensions = (
            "offset_dims = [1, 2], start_index_map = [0, 1], index_vector_dim = 0"
        )
        sizes, output = "2, 3", (3, 2, 3)
        expected = np.stack([x[1:3, 2:5], x[:2, :3], x[3:5, 3:6]])
    index_type = "tensor<" + "x".join(map(str, indices.shape)) + "xi32>"
    output_type = "tensor<" + "x".join(map(str, output)) + "xf32>"
    code = f"""module {{ func.func @main(%x: tensor<5x6xf32>, %i: {index_type}) -> {output_type} {{
      %r = "stablehlo.gather"(%x, %i) {{dimension_numbers = #stablehlo.gather<{dimensions}>,
          slice_sizes = array<i64: {sizes}>, indices_are_sorted = false}}
          : (tensor<5x6xf32>, {index_type}) -> {output_type}
      return %r : {output_type}
    }} }}"""
    executable = compile(
        code,
        (TensorSpec(x.shape), TensorSpec(indices.shape, "int32")),
        TensorSpec(output),
        backend="tvm",
    )
    actual = executable.get(executable(executable.put((x, indices))))
    np.testing.assert_array_equal(actual, expected)


def test_cropping():
    from jax import lax

    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    check(
        lambda x: lax.pad(x, np.float32(-1), ((-1, 1, 0), (1, -1, 0))),
        x,
        np.pad(x[1:, :-1], ((0, 1), (1, 0)), constant_values=-1),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "bool",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
)
def test_constants(kind):
    import jax.numpy as jnp
    import ml_dtypes

    dtype = ml_dtypes.bfloat16 if kind == "bfloat16" else np.dtype(kind)
    values = np.array([0, 1, 3, 7], dtype=dtype)
    if kind == "uint64":
        values += np.uint64(2**63)

    def model(x):
        constant = jnp.asarray(values)
        return jnp.logical_xor(x, constant) if kind == "bool" else x + constant

    expected = np.logical_xor(values, values) if kind == "bool" else values + values
    check(model, values, expected)


@pytest.mark.parametrize(
    "shape,other,dimensions,equation",
    [
        ((3,), (4,), (((), ()), ((), ())), "a,b->ab"),
        ((3,), (3,), (((0,), (0,)), ((), ())), "a,a->"),
        (
            (2, 3, 4, 5),
            (4, 5, 3, 6),
            (((1, 2), (2, 0)), ((3,), (1,))),
            "acdb,dbce->bae",
        ),
    ],
)
def test_contractions(shape, other, dimensions, equation):
    from jax import lax

    rng = np.random.default_rng(314)
    left, right = [rng.normal(size=s).astype(np.float32) for s in (shape, other)]
    check(
        lambda x: lax.dot_general(*x, dimensions),
        (left, right),
        np.einsum(equation, left, right),
    )


def test_accumulation():
    from jax import lax

    rng = np.random.default_rng(715)
    left = rng.normal(size=(3, 128)).astype(np.float16)
    right = rng.normal(size=(128, 4)).astype(np.float16)
    check(
        lambda x: lax.dot_general(
            *x, (((1,), (0,)), ((), ())), preferred_element_type=np.float32
        ),
        (left, right),
        left.astype(np.float32) @ right.astype(np.float32),
    )


def test_boolean():
    from jax import lax

    left = np.array([[True, True], [False, True]])
    right = np.array([[True, False], [True, True]])
    check(
        lambda x: lax.dot_general(*x, (((1,), (0,)), ((), ()))),
        (left, right),
        left @ right,
    )


def test_decoder():
    import jax.numpy as jnp
    from jax import lax

    def model(state):
        tokens, embedding, qkv, projection, up, down = state
        dimensions = lax.GatherDimensionNumbers(
            offset_dims=(2,), collapsed_slice_dims=(0,), start_index_map=(0,)
        )
        x = lax.gather(embedding, tokens[..., None], dimensions, (1, 8), mode="clip")
        normalized = x * lax.rsqrt(
            jnp.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5)
        )
        queries, keys, values = jnp.split(normalized @ qkv, 3, axis=-1)
        queries, keys, values = [
            v.reshape(2, 5, 2, 4).transpose(0, 2, 1, 3) for v in (queries, keys, values)
        ]
        scores = (queries @ keys.swapaxes(-1, -2)) * np.float32(0.5)
        positions = jnp.arange(5, dtype=jnp.int32)
        mask = positions[:, None] >= positions[None, :]
        scores = jnp.where(mask, scores, np.float32(-1e9))
        probabilities = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        probabilities = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
        attended = (probabilities @ values).transpose(0, 2, 1, 3).reshape(2, 5, 8)
        x = x + attended @ projection
        x = x + jnp.tanh(x @ up) @ down
        logits = x @ embedding.T
        return logits, jnp.mean(logits * logits)

    rng = np.random.default_rng(2026)
    tokens = rng.integers(0, 11, (2, 5), dtype=np.int32)
    weights = tuple(
        rng.normal(0, 0.2, shape).astype(np.float32)
        for shape in [(11, 8), (8, 24), (8, 8), (8, 16), (16, 8)]
    )
    executable = compile(model, specification((tokens, *weights)), backend="tvm")
    for perturbation in (0.0, 0.01):
        embedding, qkv, projection, up, down = tuple(
            w + np.float32(perturbation) for w in weights
        )
        x = embedding[tokens]
        normalized = x / np.sqrt(
            np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5)
        )
        queries, keys, values = [
            v.reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
            for v in np.split(normalized @ qkv, 3, axis=-1)
        ]
        scores = (queries @ keys.swapaxes(-1, -2)) * np.float32(0.5)
        scores = np.where(
            np.arange(5)[:, None] >= np.arange(5)[None, :], scores, np.float32(-1e9)
        )
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attended = (probabilities @ values).transpose(0, 2, 1, 3).reshape(2, 5, 8)
        x = x + attended @ projection
        x = x + np.tanh(x @ up) @ down
        reference = x @ embedding.T
        resident = executable.put((tokens, embedding, qkv, projection, up, down))
        actual, loss = executable.get(executable.submit(resident).wait())
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            loss, np.mean(reference * reference), rtol=2e-5, atol=2e-5
        )
