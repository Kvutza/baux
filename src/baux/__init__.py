from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._lazy import module_attr


def _extendextension() -> None:
    package_dir = Path(__file__).resolve().parent
    package_paths = globals().get("__path__")
    if package_paths is None:
        return

    for entry in sys.path:
        try:
            candidate = Path(entry) / "baux"
        except TypeError:
            continue
        if candidate == package_dir:
            continue
        if not any(candidate.glob("baux_rust*.so")) and not any(
            candidate.glob("baux_rust*.pyd")
        ):
            continue
        candidate_str = str(candidate)
        if candidate_str not in package_paths:
            package_paths.append(candidate_str)
        break


_extendextension()

try:
    from . import baux_rust as _baux  # type: ignore
except ImportError:
    _baux = None

if _baux is not None:
    sys.modules.setdefault("baux._baux", _baux)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Device": ("._rust", "Device"),
    "Tensor": ("._rust", "Tensor"),
    "Var": ("._rust", "Var"),
    "Span": ("._rust", "Span"),
    "Enn": (".enn.enn_class", "Enn"),
    "EnnParams": (".enn.enn_params", "ENNParams"),
    "ENNParams": (".enn.enn_params", "ENNParams"),
    "EnnFitter": (".enn.enn_fitter", "ENNFitter"),
    "Posterior": ("._rust", "Posterior"),
    "Region": ("._rust", "Region"),
    "Portfolio": ("._rust", "Portfolio"),
    "Trial": ("._rust", "Trial"),
    "RegionTrial": ("._rust", "RegionTrial"),
    "ENN": (".enn.enn_class", "ENN"),
    "ENNFitter": (".enn.enn_fitter", "ENNFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".bo", "create_optimizer"),
    "BayesOptimizer": (".bo", "BayesOptimizer"),
    "Optimizer": (".bo", "Optimizer"),
    "AskBatch": (".bo", "AskBatch"),
    "AskId": (".bo", "AskId"),
    "enn_optimizer": ("._rust", "enn_optimizer"),
    "create_zero": ("._rust", "create_zero"),
    "create_lhd": ("._rust", "create_lhd"),
    "Telemetry": (".bo", "Telemetry"),
    "TellPredictionError": (".bo", "TellPredictionError"),
    "OptimizerConfig": (".turbo.config", "OptimizerConfig"),
    "turbo_one": (".turbo.config", "turbo_one"),
    "turbo_zero": (".turbo.config", "turbo_zero"),
    "turbo_enn": (".turbo.config", "turbo_enn"),
    "lhd_only": (".turbo.config", "lhd_only"),
    "TurboRegion": (".region", "TurboRegion"),
    "MorboRegion": (".region", "MorboRegion"),
    "GlobalRegion": (".region", "GlobalRegion"),
    "RegionPolicy": (".region", "RegionPolicy"),
    "RegionBounds": (".region", "RegionBounds"),
    "TurboTrustRegion": (".turbo.config.trust_region", "TurboTrustRegion"),
    "MorboTrustRegion": (".turbo.config.trust_region", "MorboTrustRegion"),
    "NoTrustRegion": (".turbo.config.trust_region", "NoTrustRegion"),
    "CandidateSampler": (".turbo.config", "CandidateSampler"),
    "InitStrategy": (".turbo.config", "InitStrategy"),
    "AcquisitionKind": (".turbo.config", "AcquisitionKind"),
    "IndexKind": (".index", "IndexKind"),
    "NeighborIndex": (".index", "NeighborIndex"),
    "DistanceMetric": (".index", "DistanceMetric"),
    "Storage": (".index", "Storage"),
    "DiagonalMetric": (".surrogate", "DiagonalMetric"),
    "MetricLearning": (".surrogate", "MetricLearning"),
    "EnnSurrogate": (".surrogate", "EnnSurrogate"),
    "ENNSurrogate": (".surrogate", "ENNSurrogate"),
    "GaussianProcess": (".surrogate", "GaussianProcess"),
    "NoSurrogate": (".surrogate", "NoSurrogate"),
    "ExternalSurrogate": (".surrogate", "ExternalSurrogate"),
    "BackendCapabilities": (".runtime", "BackendCapabilities"),
    "capabilities": (".runtime", "capabilities"),
    "verify_upstream_lock": ("._baux", "verify_upstream_lock"),
}

if _baux is not None and hasattr(_baux, "verify_upstream_lock"):
    verify_upstream_lock = _baux.verify_upstream_lock


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


__all__: list[str] = [
    "ENN",
    "AcquisitionKind",
    "AskBatch",
    "AskId",
    "BackendCapabilities",
    "BayesOptimizer",
    "CandidateSampler",
    "Device",
    "DiagonalMetric",
    "DistanceMetric",
    "ENNFitter",
    "ENNParams",
    "ENNSurrogate",
    "Enn",
    "EnnFitter",
    "EnnParams",
    "EnnSurrogate",
    "ExternalSurrogate",
    "GaussianProcess",
    "GlobalRegion",
    "IndexKind",
    "InitStrategy",
    "MetricLearning",
    "MorboRegion",
    "MorboTrustRegion",
    "NeighborIndex",
    "NoSurrogate",
    "NoTrustRegion",
    "Optimizer",
    "OptimizerConfig",
    "Portfolio",
    "Posterior",
    "Region",
    "RegionBounds",
    "RegionPolicy",
    "RegionTrial",
    "Span",
    "Storage",
    "Telemetry",
    "TellPredictionError",
    "Tensor",
    "Trial",
    "TurboRegion",
    "TurboTrustRegion",
    "Var",
    "capabilities",
    "create_lhd",
    "create_optimizer",
    "create_zero",
    "enn_optimizer",
    "experimental",
    "lhd_only",
    "turbo_enn",
    "turbo_one",
    "turbo_zero",
    "verify_upstream_lock",
]
