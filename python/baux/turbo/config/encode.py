"""Serialize optimizer configurations for the native Rust engine."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .acquisition import (
    DrawAcquisitionConfig,
    ParetoAcquisitionConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .candidates import CandidateSampler, RAASPDriver
from .initialization import LHDOnlyInit
from .optimizer import OptimizerConfig
from .surrogate import (
    INDEX_NAMES,
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    NoSurrogate,
)
from .trust_region import (
    MorboTrustRegion,
    NoTrustRegion,
    RegionalSearchPolicy,
    TurboTrustRegion,
)

ENN_K = 10
_CAND_FACTOR = 100.0
_CAND_MAX = 5000


def enn_k(cfg: OptimizerConfig) -> int:
    """Resolve the neighbor count k from an ENN surrogate configuration."""
    sur = cfg.surrogate
    if not isinstance(sur, ENNSurrogate):
        raise TypeError(f"expected ENNSurrogate, got {type(sur)!r}")
    return ENN_K if sur.k is None else int(sur.k)


def _acq(cfg: OptimizerConfig) -> dict[str, Any]:
    acq = cfg.acquisition
    if isinstance(acq, UCBAcquisitionConfig):
        return {"acquisition": "ucb", "acquisition_beta": float(acq.beta)}
    if isinstance(acq, DrawAcquisitionConfig):
        return {"acquisition": "thompson"}
    if isinstance(acq, RandomAcquisitionConfig):
        return {"acquisition": "random"}
    if isinstance(acq, ParetoAcquisitionConfig):
        return {"acquisition": "pareto"}
    return {}


def _cand(cfg: OptimizerConfig) -> dict[str, Any]:
    c = cfg.candidates
    rv = {
        CandidateSampler.SOBOL: "sobol",
        CandidateSampler.UNIFORM: "uniform",
        CandidateSampler.RAASP: "raasp",
    }.get(c.sampler)
    out: dict[str, Any] = {"candidate_rv": rv} if rv else {}
    if c.num_candidates is None:
        out.update(num_candidates_factor=_CAND_FACTOR, max_candidates=_CAND_MAX)
    else:
        n = int(c.num_candidates)
        out.update(num_candidates_factor=1.0, min_candidates=n)
        if c.num_candidates_per_arm is None:
            out["max_candidates"] = n
    if c.num_candidates_per_arm is not None:
        out["num_candidates_per_arm"] = int(c.num_candidates_per_arm)
    if c.num_pert is not None:
        out["num_pert"] = int(c.num_pert)
    if c.sampler is CandidateSampler.RAASP:
        out["raasp_fast"] = c.raasp_driver is RAASPDriver.FAST
    if c.tied_dims is not None:
        out["tied_dims"] = [[int(col) for col in g] for g in c.tied_dims]
    return out


def _tr(cfg: OptimizerConfig) -> dict[str, Any]:
    tr = cfg.trust_region
    if isinstance(tr, RegionalSearchPolicy):
        return encode_regions(tr)
    if isinstance(tr, MorboTrustRegion):
        out: dict[str, Any] = {
            "trust_region": "morbo",
            "num_metrics": int(tr.num_metrics),
            "alpha": float(tr.alpha),
            "length_init": float(tr.length_init),
            "length_min": float(tr.length_min),
            "length_max": float(tr.length_max),
            "rescalarize": tr.rescalarize.value,
        }
        if tr.noise_aware:
            out["noise_aware"] = True
        return out
    if not isinstance(tr, TurboTrustRegion):
        return {}

    out = {}
    if tr.length_init != 0.8:
        out["length_init"] = float(tr.length_init)
    if abs(tr.length_min - 0.5**7) > 1e-12:
        out["length_min"] = float(tr.length_min)
    if tr.length_max != 1.6:
        out["length_max"] = float(tr.length_max)
    if tr.noise_aware:
        out["noise_aware"] = True
    return out


def _ennmetric(sur: ENNSurrogate, out: dict[str, Any]) -> None:
    ml = str(getattr(sur.metric_learning, "value", sur.metric_learning)).lower()
    if ml != "none":
        out["metric_learning"] = ml
    if sur.tied_dims is not None:
        out["tied_dims"] = [[int(c) for c in g] for g in sur.tied_dims]
    if sur.unscaled_dims is not None:
        out["unscaled_dims"] = [int(c) for c in sur.unscaled_dims]
    if sur.geometry != "euclidean":
        out["geometry"] = sur.geometry
        out["local_k"] = int(sur.local_k)
    if sur.stiefel is not None:
        loadings = getattr(sur.stiefel, "loadings", sur.stiefel)
        out["stiefel"] = np.asarray(loadings, dtype=float)
    if sur.stiefel_rank is not None:
        out["stiefel_rank"] = int(sur.stiefel_rank)
    if sur.calibrate_alpha is not None:
        out["calibrate_alpha"] = float(sur.calibrate_alpha)
    if sur.affine is not None:
        aff = sur.affine
        a, b, c = (
            (aff.a, aff.b, aff.c)
            if hasattr(aff, "a") and hasattr(aff, "b") and hasattr(aff, "c")
            else aff
        )
        out["affine"] = (
            np.asarray(a, dtype=float).ravel().tolist(),
            np.asarray(b, dtype=float).ravel().tolist(),
            np.asarray(c, dtype=float).ravel().tolist(),
        )


def _enn(cfg: OptimizerConfig) -> dict[str, Any]:
    sur = cfg.surrogate
    if not isinstance(sur, ENNSurrogate):
        return {}

    out: dict[str, Any] = {}
    if sur.index_driver in INDEX_NAMES:
        out["index_driver"] = INDEX_NAMES[sur.index_driver]
    if sur.fit.num_samples is not None:
        out["num_samples"] = int(sur.fit.num_samples)
    if sur.fit.num_candidates is not None:
        out["num_candidates"] = int(sur.fit.num_candidates)
    if sur.scale_x:
        out["scale_x"] = True
    if sur.enn_storage is not None:
        out["enn_storage"] = sur.enn_storage
    if sur.work_dir is not None:
        out["work_dir"] = os.fspath(sur.work_dir)
    if sur.y_bounds is not None:
        out["y_bounds"] = np.asarray(sur.y_bounds, dtype=float)
    _ennmetric(sur, out)
    return out


def regional_acquisition(acquisition) -> tuple[str | None, float | None]:
    """Encode a regional acquisition config into its native name and optional beta."""
    if acquisition is None:
        return None, None
    if isinstance(acquisition, UCBAcquisitionConfig):
        return "ucb", float(acquisition.beta)
    if isinstance(acquisition, DrawAcquisitionConfig):
        return "thompson", None
    if isinstance(acquisition, RandomAcquisitionConfig):
        return "random", None
    if isinstance(acquisition, ParetoAcquisitionConfig):
        return "pareto", None
    raise ValueError(f"unsupported regional acquisition: {type(acquisition)!r}")


def encode_regions(tr: RegionalSearchPolicy) -> dict[str, Any]:
    """Encode a regional search policy into native configuration parameters."""
    out = {
        "region_sharing": tr.sharing,
        "region_successes": tr.success_tolerance,
        "region_failures": tr.failure_tolerance,
        "region_allocation": tr.allocation,
        "region_restart_memory": tr.restart_memory,
        "region_restart_candidates": int(tr.restart_center.candidates),
        "region_restart_center_selection": tr.restart_center.selection,
        "region_controller": tr.controller,
        "length_init": float(tr.length_init),
        "length_min": float(tr.length_min),
        "length_max": float(tr.length_max),
    }
    if tr.acquisitions:
        acquisitions = [
            regional_acquisition(acquisition) for acquisition in tr.acquisitions
        ]
        out["region_acquisitions"] = [item[0] for item in acquisitions]
        out["region_acquisition_betas"] = [item[1] for item in acquisitions]
    return out


def encode(cfg: OptimizerConfig) -> dict[str, Any] | None:
    """Encode an OptimizerConfig into a dictionary for the native engine."""
    out = _acq(cfg) | _cand(cfg) | _tr(cfg) | _enn(cfg)
    return out or None


def supports(cfg: OptimizerConfig) -> bool:
    """Return whether the given OptimizerConfig uses a supported surrogate."""
    return isinstance(
        cfg.surrogate,
        (ENNSurrogate, GaussianProcess, NoSurrogate, ExternalSurrogate),
    )


def is_lhd(cfg: OptimizerConfig) -> bool:
    """Return whether the configuration specifies pure Latin hypercube sampling."""
    return (
        isinstance(cfg.trust_region, NoTrustRegion)
        and isinstance(cfg.init.init_strategy, LHDOnlyInit)
        and isinstance(cfg.surrogate, NoSurrogate)
    )
