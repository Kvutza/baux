"""Reduction semantics through the public compiler interface."""

import numpy as np
import pytest
from baux.experimental import TensorSpec, compile


def source(shape, axes, element, names):
    def tensor(dimensions):
        return "tensor<" + "".join(f"{n}x" for n in dimensions) + element + ">"

    count = len(names)
    scalar = tensor(())
    output = tensor([n for i, n in enumerate(shape) if i not in axes])
    initial = ", ".join(f"%i{i}: {scalar}" for i in range(count))
    inputs = ", ".join(["%x"] * count + [f"%i{i}" for i in range(count)])
    types = ", ".join([tensor(shape)] * count + [scalar] * count)
    arguments = ", ".join(f"%a{i}: {scalar}" for i in range(2 * count))
    body = "\n".join(
        f'%r{i} = "stablehlo.{name}"(%a{i}, %a{i + count}) : ({scalar}, {scalar}) -> {scalar}'
        for i, name in enumerate(names)
    )
    returns = ", ".join(f"%r{i}" for i in range(count))
    outputs = ", ".join([output] * count)
    values = ", ".join(f"%result#{i}" for i in range(count))
    dimensions = "array<i64" + (": " + ", ".join(map(str, axes)) if axes else "") + ">"
    return f"""module {{
      func.func @main(%x: {tensor(shape)}, {initial}) -> ({outputs}) {{
        %result:{count} = "stablehlo.reduce"({inputs}) ({{
          ^bb0({arguments}):
            {body}
            "stablehlo.return"({returns}) : ({", ".join([scalar] * count)}) -> ()
        }}) {{dimensions = {dimensions}}} : ({types}) -> ({outputs})
        return {values} : {outputs}
      }}
    }}"""


@pytest.mark.parametrize(
    "shape,axes",
    [((2, 3, 4), (2, 0)), ((2, 3), ()), ((2, 0, 3), (1,)), ((0, 3), (1,)), ((), ())],
)
@pytest.mark.parametrize(
    "kind,element", [("float32", "f32"), ("int32", "i32"), ("uint64", "ui64")]
)
def test_reduce(shape, axes, kind, element):
    names = ["add", "multiply", "minimum", "maximum"]
    functions = [np.add, np.multiply, np.minimum, np.maximum]
    data = (np.arange(np.prod(shape), dtype=kind) % 5 + 1).reshape(shape)
    if kind == "uint64":
        data += np.uint64(2**63)
    starts = [np.array(value, dtype=kind) for value in [7, 2, 3, 4]]
    inputs = (TensorSpec(shape, kind), *(TensorSpec((), kind) for _ in starts))
    outshape = tuple(n for i, n in enumerate(shape) if i not in axes)
    outputs = tuple(TensorSpec(outshape, kind) for _ in starts)
    executable = compile(
        source(shape, axes, element, names), inputs, outputs, backend="tvm"
    )
    actual = executable.get(executable(executable.put((data, *starts))))
    for result, initial, function in zip(actual, starts, functions, strict=True):
        expected = function.reduce(data, axis=axes, initial=initial, dtype=kind)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "names",
    [["and", "or", "xor"], ["minimum", "maximum", "xor"], ["multiply", "add", "xor"]],
)
def test_predicates(names):
    data = np.array([[True, False, True], [True, True, True]])
    starts = tuple(np.array(value) for value in [True, False, True])
    inputs = (TensorSpec(data.shape, "bool"), *(TensorSpec((), "bool") for _ in starts))
    outputs = (TensorSpec((2,), "bool"),) * 3
    executable = compile(
        source(data.shape, (1,), "i1", names), inputs, outputs, backend="tvm"
    )
    actual = executable.get(executable(executable.put((data, *starts))))
    for result, initial, function in zip(
        actual, starts, [np.logical_and, np.logical_or, np.logical_xor], strict=True
    ):
        np.testing.assert_array_equal(
            result, function.reduce(data, axis=1, initial=initial)
        )


def test_extrema():
    data = np.array(
        [[np.nan, 1, 2], [1, 2, np.nan], [np.inf] * 3, [-np.inf] * 3], np.float32
    )
    inputs = (TensorSpec(data.shape), TensorSpec(()), TensorSpec(()))
    executable = compile(
        source(data.shape, (1,), "f32", ["minimum", "maximum"]),
        inputs,
        (TensorSpec((4,)),) * 2,
        backend="tvm",
    )
    for lo, hi in [(np.inf, -np.inf), (np.nan, np.nan)]:
        args = (data, np.array(lo, np.float32), np.array(hi, np.float32))
        actual = executable.get(executable(executable.put(args)))
        for result, initial, function in zip(
            actual, args[1:], [np.minimum, np.maximum], strict=True
        ):
            np.testing.assert_array_equal(
                result, function.reduce(data, axis=1, initial=initial)
            )


@pytest.mark.parametrize(
    "kind,element", [("float16", "f16"), ("float32", "f32"), ("float64", "f64")]
)
def test_zeros(kind, element):
    data = np.array([[0.0, -0.0], [-0.0, 0.0], [-0.0, -0.0], [0.0, 0.0]], dtype=kind)
    inputs = (TensorSpec(data.shape, kind), TensorSpec((), kind), TensorSpec((), kind))
    executable = compile(
        source(data.shape, (1,), element, ["minimum", "maximum"]),
        inputs,
        (TensorSpec((4,), kind),) * 2,
        backend="tvm",
    )
    result = executable.get(
        executable(
            executable.put((data, np.array(np.inf, kind), np.array(-np.inf, kind)))
        )
    )
    np.testing.assert_array_equal(np.signbit(result[0]), [True, True, True, False])
    np.testing.assert_array_equal(np.signbit(result[1]), [False, False, True, False])
    for value in result:
        assert value.dtype == data.dtype
        np.testing.assert_array_equal(value, np.zeros(4))


def test_commutativity():
    data = np.array([1, 2, 3], np.int32)
    code = source(data.shape, (0,), "i32", ["add"]).replace(
        '"stablehlo.add"(%a0, %a1)', '"stablehlo.add"(%a1, %a0)'
    )
    executable = compile(
        code,
        (TensorSpec(data.shape, "int32"), TensorSpec((), "int32")),
        (TensorSpec((), "int32"),),
        backend="tvm",
    )
    result = executable.get(executable(executable.put((data, np.array(7, np.int32)))))
    np.testing.assert_array_equal(result[0], np.array(13, np.int32))


@pytest.mark.parametrize("body", ["subtract", "multiply"])
def test_rejected(body):
    code = source((3,), (0,), "f32", [body])
    if body == "multiply":
        code = code.replace(
            '"stablehlo.multiply"(%a0, %a1)', '"stablehlo.multiply"(%a0, %a0)'
        )
    with pytest.raises(RuntimeError, match="independent binary reducers"):
        compile(
            code, (TensorSpec((3,)), TensorSpec(())), (TensorSpec(()),), backend="tvm"
        )
