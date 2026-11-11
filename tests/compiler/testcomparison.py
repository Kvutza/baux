"""IEEE predicates against a bit-pattern oracle, including NaN payloads."""

import numpy as np
import pytest
from baux.experimental import TensorSpec, compile


@pytest.mark.parametrize("backend", ["tvm", "xla"])
@pytest.mark.parametrize(
    "kind,element,bits", [("float32", "f32", 32), ("float64", "f64", 64)]
)
def test_order(backend, kind, element, bits):
    unsigned = np.dtype(f"uint{bits}")
    sign = 1 << (bits - 1)
    infinity = np.array(np.inf, dtype=kind).view(unsigned).item()
    patterns = [
        sign | infinity | 3,
        sign | infinity,
        sign | 1,
        sign,
        0,
        1,
        infinity,
        infinity | 2,
        infinity | 3,
    ]
    values = np.array(patterns, dtype=unsigned).view(kind)
    left, right = np.broadcast_arrays(values[:, None], values[None, :])
    tensor = f"tensor<9x9x{element}>"
    boolean = "tensor<9x9xi1>"
    directions = ["EQ", "NE", "LT", "LE", "GT", "GE"]
    statements = "\n".join(
        f'%r{i} = "stablehlo.compare"(%a, %b) {{comparison_direction = #stablehlo<comparison_direction {direction}>, compare_type = #stablehlo<comparison_type TOTALORDER>}} : ({tensor}, {tensor}) -> {boolean}'
        for i, direction in enumerate(directions)
    )
    returns = ", ".join(f"%r{i}" for i in range(6))
    outputs = ", ".join([boolean] * 6)
    source = f"module {{ func.func @main(%a: {tensor}, %b: {tensor}) -> ({outputs}) {{ {statements} return {returns} : {outputs} }} }}"
    executable = compile(
        source,
        (TensorSpec((9, 9), kind),) * 2,
        (TensorSpec((9, 9), "bool"),) * 6,
        backend=backend,
    )
    actual = executable.get(executable(executable.put((left.copy(), right.copy()))))
    for value, comparison in zip(
        actual,
        [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal],
        strict=True,
    ):
        # Patterns are deliberately ordered from negative NaNs through -0/+0
        # to positive NaNs with increasing payloads.
        np.testing.assert_array_equal(
            value, comparison(np.arange(9)[:, None], np.arange(9)[None, :])
        )


def test_extrema():
    import jax.numpy as jnp

    left = np.array([np.nan, 1, 0, -0.0, 3, -np.inf], np.float32)
    right = np.array([1, np.nan, -0.0, 0, np.inf, -4], np.float32)
    executable = compile(
        lambda x: (jnp.minimum(*x), jnp.maximum(*x)),
        (TensorSpec((6,)),) * 2,
        backend="tvm",
    )
    minimum, maximum = executable.get(executable(executable.put((left, right))))
    np.testing.assert_array_equal(minimum, [np.nan, np.nan, 0, 0, 3, -np.inf])
    np.testing.assert_array_equal(maximum, [np.nan, np.nan, 0, 0, np.inf, -4])
    np.testing.assert_array_equal(np.signbit(minimum[2:4]), [True, True])
    np.testing.assert_array_equal(np.signbit(maximum[2:4]), [False, False])
