"""Checked StableHLO reducers lowered to TVM tensor expressions.

References:
    https://openxla.org/stablehlo/spec#reduce
    https://tvm.apache.org/docs/reference/api/python/te.html

StableHLO permits implementation-defined reduction trees and initializer
placement. We initialize each output once. Floating-point reassociation can
therefore differ from other compilers. Independent variadic reducers are
supported; coupled bodies such as argmax require a separate lowering.
"""

from typing import Any

from ._hlo import dtype


def _body(node: Any) -> list[str]:
    count = len(node.results)
    block = node.regions[0].blocks[0]
    ops = list(block.operations)
    supported = {"add", "multiply", "minimum", "maximum", "and", "or", "xor"}
    if len(ops) != count + 1:
        raise NotImplementedError("TVM reduce requires independent binary reducers")
    names = []
    for i, (op, returned) in enumerate(zip(ops[:-1], ops[-1].operands, strict=True)):
        name = op.operation.name.removeprefix("stablehlo.")
        pair = [block.arguments[i], block.arguments[i + count]]
        if (
            name not in supported
            or len(op.results) != 1
            or returned != op.result
            or list(op.operands) not in (pair, pair[::-1])
        ):
            raise NotImplementedError("TVM reduce requires independent binary reducers")
        names.append(name)
    return names


def _extreme(x: Any, y: Any, name: str, element: str) -> Any:
    from tvm import tirx as tir

    function = tir.min if name == "minimum" else tir.max
    integer = "uint" + element.removeprefix("bfloat").removeprefix("float")
    bits = tir.bitwise_or if name == "minimum" else tir.bitwise_and
    zero = tir.reinterpret(
        element, bits(tir.reinterpret(integer, x), tir.reinterpret(integer, y))
    )
    value = tir.if_then_else(tir.And(x == 0, y == 0), zero, function(x, y))
    # IEEE extrema propagate NaN and order negative zero below positive zero.
    return tir.if_then_else(tir.isnan(x), x, tir.if_then_else(tir.isnan(y), y, value))


def _combine(name: str, element: str) -> Any:
    from tvm import tirx as tir

    if name == "add":
        return lambda x, y: x + y
    if name == "multiply":
        return lambda x, y: x * y
    if name == "xor" and element == "bool":
        return lambda x, y: tir.Or(tir.And(x, tir.Not(y)), tir.And(tir.Not(x), y))
    if name in ("minimum", "maximum"):
        function = tir.min if name == "minimum" else tir.max
        if element.startswith(("float", "bfloat")):
            return lambda x, y: _extreme(x, y, name, element)
        return lambda x, y: function(x, y)
    functions = (
        {"and": tir.And, "or": tir.Or}
        if element == "bool"
        else {"and": tir.bitwise_and, "or": tir.bitwise_or, "xor": tir.bitwise_xor}
    )
    function = functions[name]
    return lambda x, y: function(x, y)


def _identity(name: str, element: str) -> Any:
    from tvm import tirx as tir

    if name in ("add", "or", "xor"):
        return tir.const(0, element)
    if name == "multiply":
        return tir.const(1, element)
    if name == "and":
        if element == "bool":
            return tir.const(True, element)
        return (
            tir.max_value(element)
            if element.startswith("uint")
            else tir.const(-1, element)
        )
    if element.startswith(("float", "bfloat")):
        return tir.const(float("inf") if name == "minimum" else -float("inf"), element)
    return tir.max_value(element) if name == "minimum" else tir.min_value(element)


def _tensor(
    data: Any, initial: Any, *, name: str, axes: list[int], element: str
) -> Any:
    from tvm import te

    if element == "bool":
        name = {"add": "or", "maximum": "or", "multiply": "and", "minimum": "and"}.get(
            name, name
        )
    shape = [size for i, size in enumerate(data.shape) if i not in axes]
    reduced = {i: te.reduce_axis((0, data.shape[i]), f"r{i}") for i in axes}
    combine = _combine(name, element)
    reducer = te.comm_reducer(combine, lambda kind: _identity(name, kind))
    start = initial[()].astype(element)

    def cell(*indices: Any) -> Any:
        free = iter(indices)
        index = [
            reduced[i] if i in reduced else next(free) for i in range(len(data.shape))
        ]
        value = data[tuple(index)].astype(element)
        if not axes:
            return combine(start, value)
        return reducer(value, list(reduced.values()))

    if any(int(data.shape[i]) == 0 for i in axes):
        return te.compute(shape, lambda *indices: start, name="empty_reduce")
    result = te.compute(shape, cell, name="reduce")
    if not axes:
        return result
    # The pinned TE lowering discards Reduce.init. Keep the runtime initial
    # value as an explicit expression so it participates exactly once.
    return te.compute(
        shape, lambda *indices: combine(start, result[indices]), name="initialized"
    )


def lower(importer: Any, node: Any) -> list[Any]:
    from jaxlib.mlir import ir

    names = _body(node)
    count = len(names)
    values = importer.retrieve_operands(node)
    axes = list(ir.DenseI64ArrayAttr(node.attributes["dimensions"]))
    return [
        importer.block_builder.emit_te(
            _tensor,
            values[i],
            values[i + count],
            name=name,
            axes=axes,
            element=dtype(node.results[i].type),
        )
        for i, name in enumerate(names)
    ]
