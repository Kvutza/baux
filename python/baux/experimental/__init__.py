"""Experimental native evaluators, multi-region search loops, and resident GPU sessions."""

from __future__ import annotations

import sys
from typing import Any

from .._lazy import module_attr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "compile": (".program", "compile"),
    "TensorSpec": (".program", "TensorSpec"),
    "Executable": (".program", "Executable"),
    "Invocation": (".program", "Invocation"),
    "ResidentModel": (".program", "ResidentModel"),
    "ModelOutput": (".program", "ModelOutput"),
    "ModelInvocation": (".program", "ModelInvocation"),
    "Device": (".._rust", "Device"),
    "Tensor": (".._rust", "Tensor"),
    "Var": (".._rust", "Var"),
    "Span": (".._rust", "Span"),
    "ENN": (".._rust", "ENN"),
    "Posterior": (".._rust", "Posterior"),
    "Region": (".._rust", "Region"),
    "Portfolio": (".._rust", "Portfolio"),
    "Trial": (".._rust", "Trial"),
    "RegionTrial": (".._rust", "RegionTrial"),
    "ModelPackage": (".._rust", "ModelPackage"),
    "NativeKdaModel": (".._rust", "NativeKdaModel"),
    "FlameEvaluator": (".._rust", "FlameEvaluator"),
    "MetalFlameEvaluator": (".._rust", "MetalFlameEvaluator"),
    "MetalQwenEvaluator": (".._rust", "MetalQwenEvaluator"),
    "ResidentBoSession": (".._rust", "ResidentBoSession"),
    "BayesOptimizer": (".._rust", "BayesOptimizer"),
    "Telemetry": (".._rust", "Telemetry"),
    "MultiTrustRegion": (".._rust", "MultiTrustRegion"),
    "BpannHistory": (".._rust", "BpannHistory"),
    "SharingPolicy": (".multi_region", "SharingPolicy"),
    "RegionBatch": (".multi_region", "RegionBatch"),
    "RegionCandidate": (".multi_region", "RegionCandidate"),
    "CandidateProposal": (".multi_region", "CandidateProposal"),
    "RegionRound": (".multi_region", "RegionRound"),
    "RegionalSearchLoop": (".multi_region", "RegionalSearchLoop"),
    "make_region": (".multi_region", "make_region"),
    "allocate_batches": (".multi_region", "allocate_batches"),
    "select_candidates": (".multi_region", "select_candidates"),
    "multi_region": (".multi_region", "multi_region"),
    "enn_optimizer": (".._rust", "enn_optimizer"),
    "enn_tr": (".._rust", "enn_tr"),
    "create_optimizer": (".._rust", "create_optimizer"),
    "create_zero": (".._rust", "create_zero"),
    "create_lhd": (".._rust", "create_lhd"),
    "weight_select_ucb": (".._rust", "weight_select_ucb"),
    "dense_apply": (".._rust", "dense_apply"),
    "dense_dist2": (".._rust", "dense_dist2"),
    "dense_linear": (".._rust", "dense_linear"),
    "DenseLinear": (".._rust", "DenseLinear"),
    "NeighborIndex": ("..index", "NeighborIndex"),
    "Storage": ("..index", "Storage"),
    "ENNSurrogate": ("..surrogate", "ENNSurrogate"),
}

experimental: Any = sys.modules[__name__]


_TURBO_DEFAULTS: dict[str, Any] = {
    "max_pending": 1,
    "base_variance": 0.0,
    "length_init": 0.8,
    "length_min": 0.0078125,
    "length_max": 1.6,
    "failure_tolerance": None,
    "sampler": "independent",
    "reference_seed": 0,
}


def turbo_enn(
    base: Any,
    base_value: float,
    blocks: list[Any],
    capacity: int,
    **kwargs: Any,
) -> Any:
    """Create resident CUDA BF16 search with independent sign proposals by default.

    ``sampler="independent"`` preserves legacy sign noise with TuRBO.
    ``sampler="gaussian"`` uses independent dense Gaussian noise with TuRBO and
    no reference direction. ``reference_seed`` is unused in these two modes.
    ``sampler="correlated"`` uses dense Gaussian noise and a reference direction
    initialized by ``reference_seed`` (u64), updated on acceptance. It requires
    ``max_pending=1``,
    ``failure_tolerance=None``, and ``ask(arms=1, candidates=4, ...)``. The GPU
    accepted-radius controller replaces TuRBO counters in this mode.
    ``proposals.geometry()`` reports each selected candidate index and Gaussian
    persistence (0.75 for correlated indices 0/1, otherwise 0); ``describe()``
    reports its actual selected radius. Persistence is the noise mixture
    coefficient, not measured correlation after BF16 rounding or selection.
    ``search.read_reference()`` explicitly copies the stored correlated reference
    to a host NumPy uint16 array of raw BF16 bits for validation. This is a
    model-sized copy, requires a completed round and released DLPack views, and
    errors in other modes.

    These arguments are the complete resident-search construction; there is no
    second layer of configuration precedence.
    """
    unknown = set(kwargs) - set(_TURBO_DEFAULTS)
    if unknown:
        name = sorted(unknown)[0]
        raise TypeError(f"turbo_enn() got an unexpected keyword argument {name!r}")
    merged = {**_TURBO_DEFAULTS, **kwargs}
    search_type = __getattr__("Region")
    if search_type is None:
        raise RuntimeError("turbo_enn requires the CUDA wheel")
    return search_type(
        base,
        base_value,
        blocks,
        capacity,
        **merged,
    )


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


__all__: list[str] = [
    "compile",
    "TensorSpec",
    "Invocation",
    "ResidentModel",
    "ModelOutput",
    "ModelInvocation",
    "Executable",
    "BayesOptimizer",
    "BpannHistory",
    "CandidateProposal",
    "DenseLinear",
    "Device",
    "ENN",
    "ENNSurrogate",
    "FlameEvaluator",
    "MetalFlameEvaluator",
    "MetalQwenEvaluator",
    "ModelPackage",
    "MultiTrustRegion",
    "NativeKdaModel",
    "NeighborIndex",
    "Portfolio",
    "Posterior",
    "Region",
    "RegionBatch",
    "RegionCandidate",
    "RegionRound",
    "RegionTrial",
    "RegionalSearchLoop",
    "ResidentBoSession",
    "SharingPolicy",
    "Span",
    "Storage",
    "Telemetry",
    "Tensor",
    "Trial",
    "Var",
    "allocate_batches",
    "create_lhd",
    "create_optimizer",
    "create_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "enn_optimizer",
    "enn_tr",
    "experimental",
    "make_region",
    "multi_region",
    "select_candidates",
    "turbo_enn",
    "weight_select_ucb",
]
