from __future__ import annotations

import pytest

from baux.turbo.config import (
    AcquisitionKind,
    CandidateConfig,
    CandidateSampler,
    RAASPDriver,
    SurrogateFit,
    ENNSurrogate,
    UCBAcquisitionConfig,
    turbo_enn,
)
from baux.turbo.config.encode import encode
from baux.turbo.config.surrogate import NeighborIndex


@pytest.mark.parametrize(
    ("samples", "candidates"),
    [(100, None), (None, 500), (50, 200)],
)
def test_fitparams(samples, candidates):
    cfg = turbo_enn(
        enn=ENNSurrogate(
            k=4,
            fit=SurrogateFit(
                num_samples=samples,
                num_candidates=candidates,
            ),
        )
    )
    out = encode(cfg)
    assert out is not None
    if samples is not None:
        assert out["num_samples"] == samples
    if candidates is not None:
        assert out["num_candidates"] == candidates


def test_indexdriver():
    cfg = turbo_enn(enn=ENNSurrogate(index_driver=NeighborIndex.BPANN_DISK))
    assert encode(cfg)["index_driver"] == "bpann_disk"


def test_ucbbeta():
    cfg = turbo_enn(
        acquisition=AcquisitionKind.UCB,
        enn=ENNSurrogate(fit=SurrogateFit(num_samples=8)),
    )
    out = encode(cfg)
    assert out == {
        "acquisition": "ucb",
        "acquisition_beta": UCBAcquisitionConfig().beta,
        "candidate_rv": "sobol",
        "index_driver": "exact",
        "max_candidates": 5000,
        "num_candidates_factor": 100.0,
        "num_samples": 8,
    }


def test_raaspfast():
    cfg = turbo_enn(
        candidates=CandidateConfig(
            sampler=CandidateSampler.RAASP,
            raasp_driver=RAASPDriver.FAST,
        )
    )
    assert encode(cfg)["raasp_fast"] is True
