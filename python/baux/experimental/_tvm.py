"""Compile TVM out of process; execute its artifact in the calling process.

The pinned macOS runtime combination crashes inside TVM's LLVM parser after
IREE execution. Isolate compilation, keeping resident execution free of
IPC and host transfers. This also turns compiler crashes into Python errors.

Reference: https://tvm.apache.org/docs/reference/api/python/runtime.html
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


class Tvm:
    """Relax VM retaining its loaded library, device and resident buffers."""

    def __init__(self, program: Any, device: str) -> None:
        import tvm
        from tvm.runtime.vm import VirtualMachine

        targets = {"cpu": "llvm", "metal": "metal", "cuda": "cuda", "vulkan": "vulkan"}
        if device not in targets:
            raise ValueError(f"unsupported TVM device {device!r}")
        # TVMFFI returns its core Device while the VM annotation names the
        # compatibility wrapper; they are the same runtime value.
        self.device: Any = tvm.device(device)
        if not getattr(self.device, "exist", False):
            raise RuntimeError(f"TVM device {device!r} is unavailable")
        self._storage = TemporaryDirectory(prefix="baux-tvm-")
        library = Path(self._storage.name) / "program.so"
        request = {
            "source": program.source,
            "entry": program.entry,
            "inputs": program.inputs,
            "outputs": program.outputs,
            "target": targets[device],
        }
        process = subprocess.run(
            [sys.executable, "-m", __name__, str(library)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode:
            self._storage.cleanup()
            raise RuntimeError(
                f"TVM compiler exited with {process.returncode}: {process.stderr}"
            )
        artifact = tvm.runtime.load_module(str(library))
        self.vm = VirtualMachine(artifact, self.device)

    def put(self, value: Any) -> Any:
        import tvm

        return tvm.runtime.tensor(value, self.device)

    def get(self, value: Any) -> Any:
        return value.numpy()

    def run(self, values: list[Any]) -> list[Any]:
        return list(self.vm["main"](*values))


def main() -> None:
    import tvm

    from .. import baux_rust
    from ._relax import lower

    request = json.load(sys.stdin)
    target = tvm.target.Target(request.pop("target"))
    for name in ("inputs", "outputs"):
        request[name] = [tuple(spec) for spec in request[name]]
    program = baux_rust.Program(**request)
    artifact = tvm.compile(lower(program), target=target)
    artifact.export_library(sys.argv[1])


if __name__ == "__main__":
    main()
