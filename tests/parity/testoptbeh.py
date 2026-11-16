from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import (
    AcquisitionKind,
    ENNSurrogate,
    SurrogateFit,
    turbo_enn,
    turbo_zero,
)

pytest.importorskip("baux._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def _trlengths(state, num_arms: int, num_cycles: int):
    lengths = []
    for _ in range(num_cycles):
        state, batch = baux.ask(state, num_arms=num_arms)
        y = _obj(batch.points)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)
        lengths.append(state.tr_length)
    return lengths


def test_001():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    config = turbo_enn()
    rng = np.random.default_rng(41)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))

    lengths = _trlengths(state, num_arms, num_cycles=8)
    assert len(lengths) == 8
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_002():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=6,
    )
    rng = np.random.default_rng(47)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    lengths = _trlengths(state, num_arms, num_cycles=6)
    assert len(lengths) == 6
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_003():
    from .optimizer_checks import check_opt

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.UCB,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(19)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    check_opt(state, bounds)


def test_004():
    from .optimizer_checks import check_opt

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.THOMPSON,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(23)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    check_opt(state, bounds)


def test_multiobjectiveroute():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(k=4, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(3)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, num_arms=2)
    assert batch.points.shape == (2, 2)


def test_multiobjectivewidth():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(5)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, num_arms=2)
    x = batch.points
    y = _obj(x).reshape(-1, 1)
    state, _ = baux.tell(state, x, y)
    assert x.shape == (2, 2)
    state, batch2 = baux.ask(state, num_arms=2)
    x2 = batch2.points
    y2 = np.column_stack([_obj(x2), -_obj(x2)])
    with pytest.raises(ValueError, match=r"unsupported|shape|dimension"):
        baux.tell(state, x2, y2)


def test_005():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(11)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, num_arms=2)
    x = batch.points
    y = np.column_stack([_obj(x), -_obj(x)])
    state, _ = baux.tell(state, x, y)
    state, batch2 = baux.ask(state, num_arms=2)
    x2 = batch2.points
    y2 = np.column_stack([_obj(x2), -_obj(x2)])
    state, _ = baux.tell(state, x2, y2)
    assert x2.shape == (2, 2)
    assert y2.shape == (2, 2)


def test_006():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acquisition=AcquisitionKind.PARETO,
        enn=ENNSurrogate(k=3, fit=SurrogateFit(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(7)
    state = baux.init(bounds, config=config, seed=int(rng.integers(2**63 - 1)))
    state, batch = baux.ask(state, num_arms=4)
    x = batch.points
    y = np.column_stack([_obj(x), -_obj(x)])
    state, _ = baux.tell(state, x, y)
    y_obs = state.y_obs
    assert y_obs.shape[1] == 2


def test_007():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero(num_init=4)
    rng = np.random.default_rng(7)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))

    for _ in range(25):
        state, batch = baux.ask(state, num_arms=1)
        y = _obj(batch.points).reshape(-1, 1)
        state, _ = baux.tell(state, batch.points, y)

    assert state.num_observations == 25
