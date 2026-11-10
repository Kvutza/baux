"""Trace a structured function using JAX's StableHLO lowering.

Reference: https://docs.jax.dev/en/latest/aot.html
OpTree owns structure, so custom dataclasses need only one registration.
"""

from collections.abc import Callable
from typing import Any


def lower(function: Callable[[Any], Any], inputs: Any) -> tuple[str, Any]:
    import jax

    from .program import TensorSpec, _flatten, _spec

    leaves, structure = _flatten(inputs)
    abstract = [
        jax.ShapeDtypeStruct(spec.shape, spec.dtype) for spec in map(_spec, leaves)
    ]
    output_tree = None

    def flat(*values: Any) -> tuple[Any, ...]:
        nonlocal output_tree
        result = function(structure.unflatten(values))
        outputs, output_tree = _flatten(result)
        return tuple(outputs)

    # Trace shapes, never concrete observations or parameters. Enabling x64
    # here preserves declared dtypes instead of silently narrowing them.
    with jax.enable_x64():
        lowered = jax.jit(flat, keep_unused=True).lower(*abstract)
    if output_tree is None:
        raise RuntimeError("tracing did not produce an output structure")
    specs = [TensorSpec(x.shape, str(x.dtype)) for x in lowered.out_info]
    return str(lowered.compiler_ir(dialect="stablehlo")), output_tree.unflatten(specs)
