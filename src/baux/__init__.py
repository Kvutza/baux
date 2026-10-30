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

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ENN": (".enn.enn_class", "ENN"),
    "ENNFitter": (".enn.enn_fitter", "ENNFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".bo", "create_optimizer"),
    "BayesOptimizer": (".bo", "BayesOptimizer"),
    "Optimizer": (".bo", "Optimizer"),
    "enn_optimizer": ("._rust", "enn_optimizer"),
    "create_zero": ("._rust", "create_zero"),
    "create_lhd": ("._rust", "create_lhd"),
    "Telemetry": (".bo", "Telemetry"),
    "OptimizerConfig": (".turbo.config", "OptimizerConfig"),
    "turbo_one": (".turbo.config", "turbo_one"),
    "turbo_zero": (".turbo.config", "turbo_zero"),
    "turbo_enn": (".turbo.config", "turbo_enn"),
    "lhd_only": (".turbo.config", "lhd_only"),
    "TurboTrustRegion": (".turbo.config.trust_region", "TurboTrustRegion"),
    "MorboTrustRegion": (".turbo.config.trust_region", "MorboTrustRegion"),
    "NoTrustRegion": (".turbo.config.trust_region", "NoTrustRegion"),
    "CandidateSampler": (".turbo.config", "CandidateSampler"),
    "InitStrategy": (".turbo.config", "InitStrategy"),
    "AcquisitionKind": (".turbo.config", "AcquisitionKind"),
    "TargetCapabilities": (".runtime", "TargetCapabilities"),
    "capabilities": (".runtime", "capabilities"),
}


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


__all__: list[str] = [
    "ENN",
    "AcquisitionKind",
    "TargetCapabilities",
    "BayesOptimizer",
    "CandidateSampler",
    "ENNFitter",
    "InitStrategy",
    "MorboTrustRegion",
    "NoTrustRegion",
    "Optimizer",
    "OptimizerConfig",
    "Telemetry",
    "TurboTrustRegion",
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
]
