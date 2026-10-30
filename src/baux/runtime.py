"""Capabilities of the installed Rust extension.

Operation support and build features come exclusively from Rust. Inspecting
capabilities does not initialize an accelerator unless ``probe=True`` is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

ExecutionTargetName = Literal["cpu", "cuda", "metal", "opencl"]


@dataclass(frozen=True, slots=True)
class TargetCapabilities:
    """Compiled support, target availability, and transfer requirements.

    ``available=None`` means compiled but not probed. ``direct_operations``
    describes where arithmetic executes, not memory residency. Operations in
    ``host_operations`` accept/return host arrays. ``array_frameworks`` lists
    installed Python packages only; it does not establish compiled support.
    A successful target probe exercises neighbor search, not every operation.
    """

    name: ExecutionTargetName
    compiled: bool
    available: bool | None
    reason: str | None
    direct_operations: tuple[str, ...]
    fallback_operations: tuple[str, ...]
    missing_operations: tuple[str, ...]
    host_operations: tuple[str, ...]
    array_frameworks: tuple[str, ...]


def capabilities(*, probe: bool = False) -> tuple[TargetCapabilities, ...]:
    """Inspect the installed extension; optionally exercise accelerator targets.

    Runtime probes can initialize drivers and compile kernels. Python framework
    installation or GPU detection never substitutes for compiled target support.
    ``posterior`` describes the original host-array interface.
    ``device_posterior`` describes FP32 device queries and DLPack moments/draws
    for explicit CUDA/Metal indices, scale_x=False, and unbounded outputs.
    """
    from . import baux_rust

    frameworks = tuple(name for name in ("torch", "jax") if find_spec(name) is not None)
    return tuple(
        TargetCapabilities(
            name=row["name"],
            compiled=row["compiled"],
            available=row["available"],
            reason=row["reason"],
            direct_operations=tuple(
                name for name, status in row["operations"].items() if status == "direct"
            ),
            fallback_operations=tuple(
                name
                for name, status in row["operations"].items()
                if status == "fallback"
            ),
            missing_operations=tuple(
                name
                for name, status in row["operations"].items()
                if status == "missing"
            ),
            host_operations=tuple(row["host_operations"]),
            array_frameworks=frameworks,
        )
        for row in baux_rust.capabilities(probe=probe)
    )


__all__ = ["ExecutionTargetName", "TargetCapabilities", "capabilities"]
