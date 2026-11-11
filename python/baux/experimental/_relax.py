"""TVM adapter with multiple-result support for explicit optimizer state.

Extends Apache TVM's StableHLO importer (Apache-2.0), without modifying it:
https://github.com/apache/tvm/tree/main/python/tvm/relax/frontend/stablehlo
"""

from __future__ import annotations

from typing import Any

from tvm.relax.frontend.stablehlo.stablehlo_translator import StableHLOImporter

from ._hlo import configure
from ._program import function, parse
from ._reduce import lower as reduce


class Importer(StableHLOImporter):
    """Bind SSA values directly across MLIR's Operation/OpView distinction."""

    def __init__(self) -> None:
        super().__init__()
        self.values = {}

    def retrieve_operands(self, node: Any) -> list[Any]:
        return [self.values[value] for value in node.operands]


def lower(program: Any) -> Any:
    import tvm
    from tvm import relax

    module = parse(program.source)
    fn = function(module, program.entry)
    importer = Importer()
    configure(importer)
    builder = relax.BlockBuilder()
    importer.block_builder = builder
    inputs = []
    for i, (shape, dtype) in enumerate(program.inputs):
        value = relax.Var(f"arg{i}", relax.TensorType(shape, dtype))
        inputs.append(value)
        importer._nodes[f"arg{i}"] = value
        importer.values[fn.regions[0].blocks[0].arguments[i]] = value
    with builder.function("main", params=inputs):
        with builder.dataflow():
            returned = None
            for op in fn.regions[0].blocks[0].operations:
                name = op.operation.name
                if name == "func.return":
                    values = importer.retrieve_operands(op)
                    values = [
                        v if isinstance(v, relax.Expr) else relax.const(v, dtype)
                        for v, (_, dtype) in zip(values, program.outputs, strict=True)
                    ]
                    returned = builder.emit_output(relax.Tuple(values))
                    break
                if name == "stablehlo.reduce":
                    for value, result in zip(
                        op.results, reduce(importer, op), strict=True
                    ):
                        importer.values[value] = result
                elif len(op.results) != 1 or name not in importer.convert_map:
                    raise NotImplementedError(
                        f"TVM StableHLO lowering unavailable for {name}"
                    )
                else:
                    importer.values[op.results[0]] = importer.convert_map[name](op)
            if returned is None:
                raise ValueError("entry function has no return")
        builder.emit_func_output(returned)
    result = builder.get()
    if not isinstance(result, tvm.IRModule):
        raise TypeError("TVM importer did not produce an IRModule")
    return result
