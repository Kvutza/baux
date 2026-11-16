from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import turbo_zero
from baux.turbo.config.encode import supports

pytest.importorskip("baux._rust")


def test_fixedcandidatecount():
    config = turbo_zero(num_candidates=500, num_init=4)
    assert supports(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, 1)
    assert batch.points.shape == (1, 2)


def test_001():
    config = turbo_zero(num_init=4)
    assert supports(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, 1)
    assert batch.points.shape == (1, 2)


def test_002():
    num_arms = 8
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(44)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=num_arms)
    while state.init_progress is not None:
        state, batch = baux.ask(state, num_arms=num_arms)
        x = batch.points
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        state, _ = baux.tell(state, x, y)
    state, batch = baux.ask(state, num_arms=num_arms)
    assert state.telemetry.num_candidates == expected


def test_badrowcount():
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))

    state, batch = baux.ask(state, num_arms=3)
    x = batch.points
    y = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="shape"):
        baux.tell(state, x, y)
