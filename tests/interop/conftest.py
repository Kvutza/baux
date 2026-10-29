from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def config():
    from baux import turbo_enn
    from baux.turbo.config import (
        CandidateConfig,
        SurrogateFit,
        ENNSurrogate,
    )

    return turbo_enn(
        num_init=2,
        candidates=CandidateConfig(num_candidates=16),
        enn=ENNSurrogate(k=4, fit=SurrogateFit(num_samples=8, num_candidates=8)),
    )


@pytest.fixture(autouse=True)
def wheel_origin():
    import baux

    root = os.environ.get("BAUX_TEST_WHEEL_ROOT")
    if root:
        assert Path(baux.__file__).is_relative_to(Path(root))
