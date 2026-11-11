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


class Native:
    """Shared transfers and completion protocol for Rust-owned sessions."""

    session: Any

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


class Iree(Native):
    """IREE compiler with execution owned by the shared Rust runtime."""

    def __init__(self, program: Any, device: str) -> None:
        import importlib.util
        import subprocess
        from pathlib import Path

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
        # Compiler LLVM versions must not share a process with TVM's LLVM.
        # The wheel ships a native CLI alongside its Python bindings.
        package = importlib.util.find_spec("iree.compiler")
        if package is None or package.origin is None:
            raise ImportError("install the IREE compiler extra")
        compiler = Path(package.origin).parent / "_mlir_libs" / "iree-compile"
        compiled = subprocess.run(
            [
                str(compiler),
                "-",
                "--iree-input-type=stablehlo",
                f"--iree-hal-target-backends={target}",
                *options,
            ],
            input=program.source.encode(),
            capture_output=True,
            check=False,
        )
        if compiled.returncode:
            raise ValueError(compiled.stderr.decode(errors="replace"))
        artifact = compiled.stdout
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


class Tvm(Native):
    """Native StableHLO importer, TVM compiler and CPU Relax VM."""

    def __init__(self, program: Any, device: str) -> None:
        from pathlib import Path

        from .. import baux_rust

        if device != "cpu":
            raise ValueError(f"native TVM currently supports CPU, got {device!r}")
        importer = Path(baux_rust.__file__).parent / ".dylibs" / "hlo_import"
        self.session = baux_rust.TvmSession(
            program.source, program.entry, str(importer)
        )


class Xla(Native):
    """Native XLA compiler and PJRT execution owned by the Rust runtime."""

    def __init__(self, program: Any, device: str) -> None:
        from .. import baux_rust

        if program.entry != "main":
            raise ValueError("XLA requires the exported entry to be named main")
        self.session = baux_rust.NativeSession.compile_xla(
            program.source, device, program.inputs, program.outputs
        )


def create(program: Any, backend: str, device: str) -> Any:
    if backend == "iree":
        return Iree(program, device)
    if backend == "tvm":
        return Tvm(program, device)
    if backend == "xla":
        return Xla(program, device)
    raise ValueError(f"unknown compiler backend {backend!r}")
