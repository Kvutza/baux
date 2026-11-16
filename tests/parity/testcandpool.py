from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import (
    CandidateConfig,
    ENNSurrogate,
    SurrogateFit,
    turbo_enn,
    turbo_zero,
)
from baux.turbo.config.encode import encode

from .optimizer_checks import make_optimizer  # noqa: F401

pytest.importorskip("baux._rust")


def test_perarmpool():
    config = turbo_zero(num_candidates_per_arm=1500, num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(99)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    assert config.candidates.resolve_candidates(num_dim=2, num_arms=3) == 4500
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=3)
        y = -np.sum((batch.points - 0.5) ** 2, axis=1).reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)
    state, batch = baux.ask(state, num_arms=3)
    assert state.telemetry.num_candidates == 4500
    y = -np.sum((batch.points - 0.5) ** 2, axis=1).reshape(-1, 1)
    state, _ = baux.tell(state, batch.points, y)
    state, batch2 = baux.ask(state, num_arms=8)
    assert batch2.points.shape == (8, 2)
    assert state.telemetry.num_candidates == 12000


def test_perarmisencoded():
    cfg_enn = turbo_enn(
        candidates=CandidateConfig(num_candidates_per_arm=25),
        num_init=4,
    )
    cfg_zero = turbo_zero(num_candidates_per_arm=25, num_init=4)
    for cfg in (cfg_enn, cfg_zero):
        overrides = encode(cfg)
        assert overrides is not None
        assert overrides.get("num_candidates_per_arm") == 25


def test_ennperarmpool():
    config = turbo_enn(
        candidates=CandidateConfig(num_candidates_per_arm=40),
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=8)),
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(101)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=3)
    assert expected == 200
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=3)
        y = -np.sum((batch.points - 0.5) ** 2, axis=1).reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)
    state, batch = baux.ask(state, num_arms=3)
    assert state.telemetry.num_candidates == expected


def test_fixedandperarm():
    config = turbo_zero(
        num_candidates=100,
        num_candidates_per_arm=50,
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(102)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    assert config.candidates.resolve_candidates(num_dim=2, num_arms=4) == 200
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=4)
        y = -np.sum((batch.points - 0.5) ** 2, axis=1).reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)
    state, batch = baux.ask(state, num_arms=4)
    assert state.telemetry.num_candidates == 200


def test_perarmformula():
    cfg = turbo_zero(num_candidates_per_arm=40, num_init=4)
    assert cfg.candidates.resolve_candidates(num_dim=2, num_arms=3) == 200
    assert cfg.candidates.resolve_candidates(num_dim=2, num_arms=8) == 320
    overrides = encode(cfg)
    assert overrides.get("num_candidates_per_arm") == 40
    assert overrides.get("num_candidates_factor") == 100.0
    assert overrides.get("max_candidates") == 5000


def _finishinit(state: baux.OptimizerState, num_arms: int) -> baux.OptimizerState:
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=num_arms)
        y = -np.sum((batch.points - 0.5) ** 2, axis=1).reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)
    return state


def test_defaultpoolmanyarms():
    num_arms = 25
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(103)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=num_arms)
    assert expected == 200
    state = _finishinit(state, num_arms)
    state, _ = baux.ask(state, num_arms=num_arms)
    assert state.telemetry.num_candidates == expected


def test_highdimpoolcap():
    num_dim = 60
    num_arms = 1
    config = turbo_zero(num_candidates_per_arm=40, num_init=4)
    bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (num_dim, 1))
    rng = np.random.default_rng(104)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    expected = config.candidates.resolve_candidates(num_dim=num_dim, num_arms=num_arms)
    assert expected == 5000
    state = _finishinit(state, num_arms)
    state, _ = baux.ask(state, num_arms=num_arms)
    assert state.telemetry.num_candidates == expected
