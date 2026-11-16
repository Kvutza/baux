"""baux Bayesian optimization, Epistemic Nearest Neighbors, and functional search API."""

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
    "Affine": (".pytree", "Affine"),
    "AskBatch": (".bo", "AskBatch"),
    "AskId": (".bo", "AskId"),
    "Categorical": (".pytree", "Categorical"),
    "Device": ("._rust", "Device"),
    "DistanceMetric": (".index", "DistanceMetric"),
    "ENNParams": (".enn.enn_params", "ENNParams"),
    "ENNState": (".enn.enn_state", "ENNState"),
    "Key": ("._rust", "Key"),
    "Log": (".pytree", "Log"),
    "NeighborIndex": (".index", "NeighborIndex"),
    "OptimizerConfig": (".turbo.config", "OptimizerConfig"),
    "OptimizerState": (".state", "OptimizerState"),
    "Periodic": (".pytree", "Periodic"),
    "Portfolio": ("._rust", "Portfolio"),
    "PyTreeSpec": (".pytree", "PyTreeSpec"),
    "Region": ("._rust", "Region"),
    "RegionTrial": ("._rust", "RegionTrial"),
    "RegionalSearchPolicy": (".region", "RegionalSearchPolicy"),
    "Simplex": (".pytree", "Simplex"),
    "Space": (".pytree", "Space"),
    "Stiefel": (".pytree", "Stiefel"),
    "Telemetry": (".bo", "Telemetry"),
    "Tensor": ("._rust", "Tensor"),
    "Transform": (".transforms", "Transform"),
    "Trial": ("._rust", "Trial"),
    "TrustRegionBounds": (".region", "TrustRegionBounds"),
    "Var": ("._rust", "Var"),
    "append": (".enn.enn_state", "append"),
    "ask": (".functional", "ask"),
    "calibrate": (".enn.enn_state", "calibrate"),
    "cancel": (".functional", "cancel"),
    "capabilities": (".runtime", "capabilities"),
    "chain": (".transforms", "chain"),
    "complete": (".functional", "complete"),
    "digest": (".state", "digest"),
    "enn": (".enn", "enn"),
    "enn_init": (".enn.enn_state", "enn_init"),
    "ensure_sync": (".enn.enn_state", "ensure_sync"),
    "experimental": (".experimental", "experimental"),
    "fit": (".enn.enn_state", "fit"),
    "fold": (".functional", "fold"),
    "freeze": (".state", "freeze"),
    "gp": (".transforms", "gp"),
    "import_observations": (".functional", "import_observations"),
    "init": (".functional", "init"),
    "key": (".functional", "key"),
    "lhd": (".transforms", "lhd"),
    "pareto": (".transforms", "pareto"),
    "persist_index": (".enn.enn_state", "persist_index"),
    "pmap": (".functional", "pmap"),
    "posterior": (".enn.enn_state", "posterior"),
    "posterior_mu": (".functional", "posterior_mu"),
    "pytree_bounds": (".pytree", "pytree_bounds"),
    "raasp": (".transforms", "raasp"),
    "random_acq": (".transforms", "random_acq"),
    "ravel": (".pytree", "ravel"),
    "regions": (".transforms", "regions"),
    "restart_region": (".functional", "restart_region"),
    "sample": (".enn.enn_state", "sample"),
    "scan": (".functional", "scan"),
    "sobol": (".transforms", "sobol"),
    "split": (".functional", "split"),
    "stein": (".enn.enn_state", "stein"),
    "step": (".functional", "step"),
    "tell": (".functional", "tell"),
    "thaw": (".state", "thaw"),
    "thompson": (".transforms", "thompson"),
    "tr_bounds": (".transforms", "tr_bounds"),
    "ucb": (".transforms", "ucb"),
    "unflatten": (".pytree", "unflatten"),
    "uniform_pert": (".transforms", "uniform_pert"),
    "with_geometry": (".enn.enn_state", "with_geometry"),
    "zero": (".transforms", "zero"),
}


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


__all__: list[str] = [
    "Affine",
    "AskBatch",
    "AskId",
    "Categorical",
    "Device",
    "DistanceMetric",
    "ENNParams",
    "ENNState",
    "Key",
    "Log",
    "NeighborIndex",
    "OptimizerConfig",
    "OptimizerState",
    "Periodic",
    "Portfolio",
    "PyTreeSpec",
    "Region",
    "RegionTrial",
    "RegionalSearchPolicy",
    "Simplex",
    "Space",
    "Stiefel",
    "Telemetry",
    "Tensor",
    "Transform",
    "Trial",
    "TrustRegionBounds",
    "Var",
    "append",
    "ask",
    "calibrate",
    "cancel",
    "capabilities",
    "chain",
    "complete",
    "digest",
    "enn",
    "enn_init",
    "ensure_sync",
    "experimental",
    "fit",
    "fold",
    "freeze",
    "gp",
    "import_observations",
    "init",
    "key",
    "lhd",
    "pareto",
    "persist_index",
    "pmap",
    "posterior",
    "posterior_mu",
    "pytree_bounds",
    "raasp",
    "random_acq",
    "ravel",
    "regions",
    "restart_region",
    "sample",
    "scan",
    "sobol",
    "split",
    "stein",
    "step",
    "tell",
    "thaw",
    "thompson",
    "tr_bounds",
    "ucb",
    "unflatten",
    "uniform_pert",
    "with_geometry",
    "zero",
]
