from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.state import OptimizerState
from baux.turbo.config import (
    ENNSurrogate,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)
from baux.turbo.config.encode import ENN_K, enn_k, supports

pytest.importorskip("baux._rust")

BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "cfg",
    [turbo_enn(), turbo_zero(), lhd_only(), turbo_one()],
)
def test_001(cfg):
    assert supports(cfg)
    rng = np.random.default_rng(0)
    opt = baux.init(BOUNDS, cfg, seed=int(rng.integers(2**63 - 1)))
    assert isinstance(opt, OptimizerState)


def test_defaultennk():
    cfg = turbo_enn(enn=ENNSurrogate(k=None))
    assert enn_k(cfg) == ENN_K
