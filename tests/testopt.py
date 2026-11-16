from __future__ import annotations

import baux
import numpy as np
import pytest
from baux.turbo.config import turbo_enn, turbo_one, turbo_zero


def _score(x: np.ndarray) -> np.ndarray:
    return -np.sum(x**2, axis=1)


@pytest.mark.parametrize("cfg", [turbo_zero(), turbo_enn()])
def test_asktellcycle(cfg):
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    state = baux.init(bounds, config=cfg, seed=0)
    state, batch = baux.ask(state, 4)
    x = batch.points
    state, y = baux.tell(state, batch, _score(x))

    assert x.shape == (4, 2)
    assert y.shape == (4,)
    state, batch2 = baux.ask(state, 2)
    assert batch2.points.shape == (2, 2)


def test_001():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    state = baux.init(bounds, config=turbo_zero(), seed=0)
    state, batch = baux.ask(state, 4)
    x = batch.points
    y = _score(x)

    assert not x.flags.writeable
    assert y.flags.writeable

    state, mu = baux.tell(state, batch, y)

    # Verify that the arrays are read-only
    assert not x.flags.writeable
    assert not mu.flags.writeable

    with pytest.raises(ValueError):
        x[0, 0] = 999.0
    with pytest.raises(ValueError):
        mu[0] = 999.0


def test_listbounds():
    state = baux.init(
        [[0.0, 1.0], [0.0, 1.0]],
        config=turbo_zero(),
        seed=1,
    )
    state, batch = baux.ask(state, 2)
    assert batch.points.shape == (2, 2)


def test_badbatchisrejected():
    state = baux.init(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=turbo_zero(),
        seed=2,
    )
    state, batch = baux.ask(state, 3)
    with pytest.raises(ValueError, match="shape"):
        baux.tell(state, batch, np.ones(2))


@pytest.mark.gp
def test_gpnoisecycle():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    state = baux.init(
        bounds,
        config=turbo_one(num_init=2),
        seed=3,
    )
    state, batch = baux.ask(state, 2)
    x = batch.points
    state, _ = baux.tell(state, batch, _score(x), y_var=np.full(2, 0.05))
    state, batch2 = baux.ask(state, 2)
    assert batch2.points.shape == (2, 2)
