from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import (
    OptimizerConfig,
    turbo_enn,
    turbo_one,
    turbo_zero,
)
from enn_helpers import sphere_objective


def _unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


@pytest.mark.parametrize(
    "config",
    [
        turbo_zero(),
        turbo_enn(),
        pytest.param(turbo_one(), marks=pytest.mark.slow),
    ],
    ids=["TURBO_ZERO", "TURBO_ENN", "TURBO_ONE"],
)
def test_001(config: OptimizerConfig) -> None:
    bounds1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    bounds2 = np.array([[2.0, 4.0], [-3.0, 1.0]], dtype=float)
    num_arms, num_steps = 2, 2
    state1 = baux.init(bounds1, config=config, seed=0)
    state2 = baux.init(bounds2, config=config, seed=0)
    for _ in range(num_steps):
        state1, batch1 = baux.ask(state1, num_arms)
        state2, batch2 = baux.ask(state2, num_arms)
        x1, x2 = batch1.points, batch2.points
        u1, u2 = _unit(x1, bounds1), _unit(x2, bounds2)
        assert np.allclose(u1, u2)
        y1 = sphere_objective(2.0 * u1 - 1.0)
        y2 = sphere_objective(2.0 * u2 - 1.0)
        assert np.allclose(y1, y2)
        state1, _ = baux.tell(state1, batch1, y1)
        state2, _ = baux.tell(state2, batch2, y2)


@pytest.mark.parametrize(
    "config",
    [turbo_zero(), turbo_enn()],
    ids=["TURBO_ZERO", "TURBO_ENN"],
)
def test_002(config: OptimizerConfig) -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms, num_steps = 2, 2

    def run_transform(scale: float, shift: float) -> np.ndarray:
        state = baux.init(bounds, config=config, seed=0)
        unit_trajectory = []
        for _ in range(num_steps):
            state, batch = baux.ask(state, num_arms)
            x = batch.points
            u = x.copy()
            base_y = sphere_objective(2.0 * u - 1.0)
            state, _ = baux.tell(state, batch, scale * base_y + shift)
            unit_trajectory.append(u)
        return np.stack(unit_trajectory, axis=0)

    traj_base = run_transform(scale=1.0, shift=0.0)
    traj_affine = run_transform(scale=2.0, shift=0.5)
    assert np.allclose(traj_base, traj_affine)
