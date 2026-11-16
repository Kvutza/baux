from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import (
    AcquisitionKind,
    CandidateConfig,
    ENNSurrogate,
    MorboTrustRegion,
    MultiObjectiveConfig,
    SurrogateFit,
    turbo_enn,
)
from morbo_objectives import separable_objective

pytest.importorskip("baux._rust")


def test_morbopostinit():
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(
            k=4,
            fit=SurrogateFit(num_samples=4, num_candidates=8),
        ),
        trust_region=MorboTrustRegion(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
            noise_aware=True,
        ),
        num_init=8,
        candidates=CandidateConfig(num_candidates=64),
    )
    rng = np.random.default_rng(43)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    num_arms = 4
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=num_arms)
        x = batch.points
        assert x.shape == (num_arms, 2)
        y = separable_objective(x)
        assert y.shape == (num_arms, 2)
        state, _ = baux.tell(state, x, y)
    state, batch = baux.ask(state, num_arms=num_arms)
    x = batch.points
    y = separable_objective(x)
    state, _ = baux.tell(state, x, y)
    assert state.telemetry.num_candidates == 64
    assert int(state.num_observations) >= num_arms


def test_morboimproves():
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(
            k=4,
            fit=SurrogateFit(num_samples=4, num_candidates=8),
        ),
        trust_region=MorboTrustRegion(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
            noise_aware=True,
        ),
        num_init=8,
        candidates=CandidateConfig(num_candidates=64),
    )
    rng = np.random.default_rng(42)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    best_y = np.array([-np.inf, -np.inf], dtype=float)
    num_rounds = 35
    num_arms = 4
    for _ in range(num_rounds):
        state, batch = baux.ask(state, num_arms=num_arms)
        x = batch.points
        assert x.shape == (num_arms, 2)
        y = separable_objective(x)
        state, _ = baux.tell(state, x, y)
        best_y = np.maximum(best_y, y.max(axis=0))
        assert np.isfinite(state.tr_length)
    assert float(best_y[0]) >= 499_000.0
    assert float(best_y[1]) >= 11.0
