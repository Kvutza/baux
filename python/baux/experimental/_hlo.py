"""StableHLO dimension semantics missing from TVM's upstream importer.

Reference: https://openxla.org/stablehlo/spec#dot_general and
https://openxla.org/stablehlo/spec#broadcast_in_dim.
"""

from math import prod
from typing import Any


def dtype(value: Any) -> str:
    from jaxlib.mlir import ir

    from ._program import _DTYPES

    element = str(ir.RankedTensorType(value).element_type)
    if element not in _DTYPES:
        raise NotImplementedError(f"TVM dtype lowering unavailable for {element}")
    return _DTYPES[element]


def constant(importer: Any, node: Any) -> Any:
    import numpy as np
    from tvm import relax

    return relax.const(np.asarray(node.attributes["value"]), dtype(node.result.type))


def broadcast(importer: Any, node: Any) -> Any:
    from jaxlib.mlir import ir
    from tvm import relax

    data = importer.retrieve_operands(node)[0]
    original = importer.get_shape(node.operands[0].type)
    shape = importer.get_shape(node.result.type)
    axes = list(ir.DenseI64ArrayAttr(node.attributes["broadcast_dimensions"]))
    expanded = [1] * len(shape)
    for axis, size in zip(axes, original, strict=True):
        expanded[axis] = size
    # TVM's annotation models a one-element tuple although Relax accepts an
    # arbitrary-rank shape here.
    expanded_shape: Any = tuple(expanded)
    data = relax.op.reshape(data, expanded_shape)
    return importer.block_builder.emit(relax.op.broadcast_to(data, shape))


def dot(importer: Any, node: Any) -> Any:
    from jaxlib.mlir.dialects import stablehlo
    from tvm import relax

    if "algorithm" in node.attributes:
        raise NotImplementedError("TVM explicit dot algorithms are not supported")
    dims = stablehlo.DotDimensionNumbers(node.attributes["dot_dimension_numbers"])
    batch = [dims.lhs_batching_dimensions, dims.rhs_batching_dimensions]
    contract = [dims.lhs_contracting_dimensions, dims.rhs_contracting_dimensions]
    shapes = [importer.get_shape(x.type) for x in node.operands]
    free = [
        [i for i in range(len(shape)) if i not in b and i not in c]
        for shape, b, c in zip(shapes, batch, contract, strict=True)
    ]
    values = importer.retrieve_operands(node)
    matrices = []
    element = dtype(node.result.type)
    for i, (value, shape, b, c, f) in enumerate(
        zip(values, shapes, batch, contract, free, strict=True)
    ):
        axes = b + (f + c if i == 0 else c + f)
        rows, cols = (f, c) if i == 0 else (c, f)
        packed = tuple(
            [shape[j] for j in b]
            + [
                prod(shape[j] for j in rows),
                prod(shape[j] for j in cols),
            ]
        )
        value = relax.op.permute_dims(value, axes)
        value = relax.op.astype(value, element)
        matrices.append(relax.op.reshape(value, packed))
    product = relax.op.matmul(*matrices, out_dtype=element)
    return importer.block_builder.emit(
        relax.op.reshape(product, importer.get_shape(node.result.type))
    )


def configure(importer: Any) -> None:
    from tvm import relax

    # Remove upstream handlers that assume special cases. Reduce is handled
    # separately because it may return multiple tensors.
    for name in ["convolution", "reduce", "reduce_window", "round_nearest_afz"]:
        importer.convert_map.pop("stablehlo." + name, None)
    importer.convert_map["stablehlo.broadcast_in_dim"] = lambda op: broadcast(
        importer, op
    )
    importer.convert_map["stablehlo.dot_general"] = lambda op: dot(importer, op)
    importer.convert_map["stablehlo.constant"] = lambda op: constant(importer, op)
    importer.convert_map["stablehlo.tanh"] = lambda op: importer.block_builder.emit(
        relax.op.tanh(importer.retrieve_operands(op)[0])
    )
