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
    "Device": ("._rust", "Device"),
    "Tensor": ("._rust", "Tensor"),
    "Var": ("._rust", "Var"),
    "Span": ("._rust", "Span"),
    "ENNParams": (".enn.enn_params", "ENNParams"),
    "Posterior": ("._rust", "Posterior"),
    "Region": ("._rust", "Region"),
    "Portfolio": ("._rust", "Portfolio"),
    "Trial": ("._rust", "Trial"),
    "RegionTrial": ("._rust", "RegionTrial"),
    "ENN": (".enn.enn_class", "ENN"),
    "ENNFitter": (".enn.enn_fitter", "ENNFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".bo", "create_optimizer"),
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
    "RegionalSearchPolicy": (".region", "RegionalSearchPolicy"),
    "TrustRegionBounds": (".region", "TrustRegionBounds"),
    "TurboTrustRegion": (".turbo.config.trust_region", "TurboTrustRegion"),
    "MorboTrustRegion": (".turbo.config.trust_region", "MorboTrustRegion"),
    "NoTrustRegion": (".turbo.config.trust_region", "NoTrustRegion"),
    "CandidateSampler": (".turbo.config", "CandidateSampler"),
    "InitStrategy": (".turbo.config", "InitStrategy"),
    "AcquisitionKind": (".turbo.config", "AcquisitionKind"),
    "NeighborIndex": (".index", "NeighborIndex"),
    "DistanceMetric": (".index", "DistanceMetric"),
    "Storage": (".index", "Storage"),
    "DiagonalMetric": (".surrogate", "DiagonalMetric"),
    "MetricLearning": (".surrogate", "MetricLearning"),
    "ENNSurrogate": (".surrogate", "ENNSurrogate"),
    "GaussianProcess": (".surrogate", "GaussianProcess"),
    "NoSurrogate": (".surrogate", "NoSurrogate"),
    "ExternalSurrogate": (".surrogate", "ExternalSurrogate"),
    "BackendCapabilities": (".runtime", "BackendCapabilities"),
    "capabilities": (".runtime", "capabilities"),
    "verify_upstream_lock": ("._rust", "verify_upstream_lock"),
    "LeafSpec": (".pytree", "LeafSpec"),
    "PyTreeSpec": (".pytree", "PyTreeSpec"),
    "pytree_bounds": (".pytree", "pytree_bounds"),
    "mlp_spec": (".pytree", "mlp_spec"),
    "mlp_forward": (".pytree", "mlp_forward"),
    "ravel": (".pytree", "ravel"),
    "unflatten": (".pytree", "unflatten"),
    "unflatten_batch": (".pytree", "unflatten_batch"),
    "unflatten_stacked": (".pytree", "unflatten_stacked"),
    "Transform": (".transforms", "Transform"),
    "chain": (".transforms", "chain"),
    "lhd": (".transforms", "lhd"),
    "raasp": (".transforms", "raasp"),
    "sobol": (".transforms", "sobol"),
    "uniform_pert": (".transforms", "uniform_pert"),
    "enn": (".enn", "enn"),
    "gp": (".transforms", "gp"),
    "zero": (".transforms", "zero"),
    "ucb": (".transforms", "ucb"),
    "thompson": (".transforms", "thompson"),
    "pareto": (".transforms", "pareto"),
    "random_acq": (".transforms", "random_acq"),
    "tr_bounds": (".transforms", "tr_bounds"),
    "regions": (".transforms", "regions"),
    "ENNState": (".functional", "ENNState"),
    "OptimizerState": (".functional", "OptimizerState"),
    "init": (".functional", "init"),
    "ask": (".functional", "ask"),
    "tell": (".functional", "tell"),
    "complete": (".functional", "complete"),
    "cancel": (".functional", "cancel"),
    "restart_region": (".functional", "restart_region"),
    "import_observations": (".functional", "import_observations"),
    "posterior_mu": (".functional", "posterior_mu"),
    "step": (".functional", "step"),
    "scan": (".functional", "scan"),
}


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


__all__: list[str] = [
    "ENN",
    "AcquisitionKind",
    "AskBatch",
    "AskId",
    "BackendCapabilities",
    "CandidateSampler",
    "Device",
    "DiagonalMetric",
    "DistanceMetric",
    "ENNFitter",
    "ENNParams",
    "ENNState",
    "ENNSurrogate",
    "ExternalSurrogate",
    "GaussianProcess",
    "InitStrategy",
    "LeafSpec",
    "MetricLearning",
    "MorboTrustRegion",
    "NeighborIndex",
    "NoSurrogate",
    "NoTrustRegion",
    "Optimizer",
    "OptimizerConfig",
    "OptimizerState",
    "Portfolio",
    "Posterior",
    "PyTreeSpec",
    "Region",
    "RegionTrial",
    "RegionalSearchPolicy",
    "Span",
    "Storage",
    "Telemetry",
    "TellPredictionError",
    "Tensor",
    "Transform",
    "Trial",
    "TrustRegionBounds",
    "TurboTrustRegion",
    "Var",
    "ask",
    "cancel",
    "capabilities",
    "chain",
    "complete",
    "create_lhd",
    "create_optimizer",
    "create_zero",
    "enn",
    "enn_optimizer",
    "experimental",
    "gp",
    "import_observations",
    "init",
    "lhd",
    "lhd_only",
    "mlp_forward",
    "mlp_spec",
    "pareto",
    "posterior_mu",
    "pytree_bounds",
    "raasp",
    "random_acq",
    "ravel",
    "regions",
    "restart_region",
    "scan",
    "sobol",
    "step",
    "tell",
    "thompson",
    "tr_bounds",
    "turbo_enn",
    "turbo_one",
    "turbo_zero",
    "ucb",
    "unflatten",
    "unflatten_batch",
    "unflatten_stacked",
    "uniform_pert",
    "verify_upstream_lock",
    "zero",
]
