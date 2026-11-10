"""Compiler adapters; dependencies are loaded only for the selected backend."""

from __future__ import annotations

from typing import Any

import numpy as np

_DTYPES = {
    "i1": "bool",
    "i32": "int32",
    "i64": "int64",
    "ui32": "uint32",
    "ui64": "uint64",
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
}


def parse(source: str) -> Any:
    from jax._src.interpreters import mlir

    context = mlir.make_ir_context()
    return mlir.ir.Module.parse(source, context)


def function(module: Any, entry: str) -> Any:
    from jaxlib.mlir import ir

    functions = [
        op
        for op in module.body.operations
        if op.operation.name == "func.func"
        and ir.StringAttr(op.attributes["sym_name"]).value == entry
    ]
    if len(functions) != 1:
        raise ValueError(f"expected one entry function {entry!r}")
    return functions[0]


def verify(program: Any) -> None:
    from jaxlib.mlir import ir

    module = parse(program.source)
    if not module.operation.verify():
        raise ValueError("StableHLO verification failed")
    fn = function(module, program.entry)
    kind = ir.FunctionType(ir.TypeAttr(fn.attributes["function_type"]).value)
    for types, expected in [
        (kind.inputs, program.inputs),
        (kind.results, program.outputs),
    ]:
        actual = []
        for value in types:
            tensor = ir.RankedTensorType(value)
            dtype = _DTYPES.get(str(tensor.element_type))
            actual.append((list(tensor.shape), dtype))
        if actual != expected:
            raise ValueError(
                f"StableHLO signature {actual} differs from declared {expected}"
            )


class Iree:
    """IREE compiler with execution owned by the shared Rust runtime."""

    def __init__(self, program: Any, device: str) -> None:
        import iree.compiler as compiler
        from jaxlib.mlir import ir

        from .. import baux_rust

        targets = {
            "cpu": "llvm-cpu",
            "metal": "metal-spirv",
        }
        if device not in targets:
            raise ValueError(f"unsupported IREE device {device!r}")
        target = targets[device]
        options = [
            "--iree-input-demote-f64-to-f32=false",
            "--iree-input-demote-f32-to-f16=false",
            "--iree-input-demote-i64-to-i32=false",
            "--iree-input-promote-bf16-to-f32=false",
            "--iree-input-promote-f16-to-f32=false",
        ]
        if device == "metal":
            # Compile embedded MSL at module load without Apple's separate
            # offline Metal toolchain.
            options.append("--iree-metal-compile-to-metallib=false")
        artifact = compiler.compile_str(
            program.source,
            target_backends=[target],
            input_type="stablehlo",
            extra_args=options,
        )
        module = parse(program.source)
        symbol = (
            ir.StringAttr(module.operation.attributes["sym_name"]).value
            if "sym_name" in module.operation.attributes
            else "module"
        )
        self.session = baux_rust.NativeSession(
            artifact,
            f"{symbol}.{program.entry}",
            device,
            program.inputs,
            program.outputs,
        )

    def put(self, value: Any) -> Any:
        return self.session.put(list(value.shape), str(value.dtype), value.tobytes())

    def get(self, value: Any) -> Any:
        return (
            np.frombuffer(value.read(), dtype=value.dtype).reshape(value.shape).copy()
        )

    def run(self, values: list[Any]) -> list[Any]:
        return self.session.run(values)

    def submit(self, values: list[Any]) -> Any:
        return self.session.submit(values)


class Xla:
    """Compile StableHLO directly through the installed PJRT client."""

    def __init__(self, program: Any, device: str) -> None:
        from jax._src import xla_bridge
        from jaxlib import xla_client

        if program.entry != "main":
            raise ValueError("XLA requires the exported entry to be named main")
        self.client = xla_bridge.get_backend(device)
        self.device = self.client.local_devices()[0]
        self.call = self.client.compile_and_load(
            program.source, [self.device], xla_client.CompileOptions()
        )

    def put(self, value: Any) -> Any:
        import jax

        with jax.enable_x64():
            return jax.device_put(value, self.device)

    def get(self, value: Any) -> Any:
        return np.array(value, copy=True)

    def run(self, values: list[Any]) -> list[Any]:
        return list(self.call.execute(values))


class QueuedAdapter:
    """Rust-owned queue around a backend's Python execution object.

    XLA and TVM compilation, buffer conversion, and execution still use their
    Python bindings. One Rust worker serializes calls, and completions have the
    same consumable ownership contract as the direct IREE runtime.
    """

    def __init__(self, engine: Any) -> None:
        from .. import baux_rust

        self.engine = engine
        self.session = baux_rust.BackendQueue(engine)

    def put(self, value: Any) -> Any:
        return self.engine.put(value)

    def get(self, value: Any) -> Any:
        return self.engine.get(value)

    def run(self, values: list[Any]) -> list[Any]:
        return self.session.run(values)

    def submit(self, values: list[Any]) -> Any:
        return self.session.submit(values)


def create(program: Any, backend: str, device: str) -> Any:
    if backend == "iree":
        return Iree(program, device)
    if backend == "xla":
        return QueuedAdapter(Xla(program, device))
    if backend == "tvm":
        from ._tvm import Tvm

        return QueuedAdapter(Tvm(program, device))
    raise ValueError(f"unknown compiler backend {backend!r}")
