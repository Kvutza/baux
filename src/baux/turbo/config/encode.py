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
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    INDEX_NAMES,
    NoSurrogate,
)
from .trust_region import (
    MorboTrustRegion,
    MultiTrustRegionPolicy,
    NoTrustRegion,
    TurboTrustRegion,
)

ENN_K = 10
_CAND_FACTOR = 100.0
_CAND_MAX = 5000


def enn_k(cfg: OptimizerConfig) -> int:
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
    return out


def _tr(cfg: OptimizerConfig) -> dict[str, Any]:
    tr = cfg.trust_region
    if isinstance(tr, MultiTrustRegionPolicy):
        return encode_multitr(tr)
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
    return out


def regional_acquisition(acquisition) -> tuple[str | None, float | None]:
    if acquisition is None:
        return None, None
    if isinstance(acquisition, UCBAcquisitionConfig):
        return "ucb", float(acquisition.beta)
    if isinstance(acquisition, DrawAcquisitionConfig):
        return "thompson", None
    if isinstance(acquisition, RandomAcquisitionConfig):
        return "random", None
    raise ValueError(f"unsupported regional acquisition: {type(acquisition)!r}")


def encode_multitr(tr: MultiTrustRegionPolicy) -> dict[str, Any]:
    out = {
        "region_sharing": tr.sharing_policy,
        "region_successes": tr.succ_tolerance,
        "region_failures": tr.fail_tolerance,
        "region_allocation": tr.allocation_policy,
        "region_restart_memory": tr.restart_memory_policy,
        "region_restart_candidates": int(tr.restart_center.candidates),
        "region_restart_center_selection": tr.restart_center.selection,
        "region_method": tr.region_method,
        "length_init": float(tr.length_init),
        "length_min": float(tr.length_min),
        "length_max": float(tr.length_max),
    }
    if tr.region_acquisitions:
        acquisitions = [
            regional_acquisition(acquisition) for acquisition in tr.region_acquisitions
        ]
        out["region_acquisitions"] = [item[0] for item in acquisitions]
        out["region_acquisition_betas"] = [item[1] for item in acquisitions]
    return out


def encode(cfg: OptimizerConfig) -> dict[str, Any] | None:
    out = _acq(cfg) | _cand(cfg) | _tr(cfg) | _enn(cfg)
    return out or None


def supports(cfg: OptimizerConfig) -> bool:
    return isinstance(
        cfg.surrogate,
        (ENNSurrogate, GaussianProcess, NoSurrogate, ExternalSurrogate),
    )


def is_lhd(cfg: OptimizerConfig) -> bool:
    return (
        isinstance(cfg.trust_region, NoTrustRegion)
        and isinstance(cfg.init.init_strategy, LHDOnlyInit)
        and isinstance(cfg.surrogate, NoSurrogate)
    )
