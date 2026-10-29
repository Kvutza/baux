from __future__ import annotations

import inspect

import numpy as np
import pytest

from baux import Telemetry, create_optimizer, turbo_enn, turbo_zero
from baux.bo import Optimizer
from baux.turbo.config import lhd_only


def test_factorysignature():
    assert set(inspect.signature(create_optimizer).parameters) == {
        "bounds",
        "config",
        "rng",
    }


@pytest.mark.parametrize(
    "cfg",
    [
        turbo_enn(num_init=2),
        turbo_zero(num_init=2),
        lhd_only(num_init=2),
    ],
)
def test_optimizercontract(cfg):
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(42))
    x = opt.ask(2)

    assert isinstance(opt, Optimizer)
    assert x.shape == (2, 2)
    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert isinstance(opt.telemetry(), Telemetry)


def test_surrogate():
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    enn = create_optimizer(
        bounds=bounds,
        config=turbo_enn(num_init=2),
        rng=np.random.default_rng(42),
    )
    zero = create_optimizer(
        bounds=bounds,
        config=turbo_zero(num_init=2),
        rng=np.random.default_rng(42),
    )

    assert enn._inner.has_surrogate()
    assert not zero._inner.has_surrogate()
    x = enn.ask(2)
    estimate = enn.tell(x, -np.sum(x**2, axis=1))
    assert estimate.shape == (2,)
    assert np.all(np.isfinite(estimate))
    assert np.allclose(np.asarray(enn._xobs.view()), x)


def test_001():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_zero(),
        rng=np.random.default_rng(0),
    )
    x = opt.ask(4)
    y = -np.sum(x**2, axis=1)

    assert x.flags.writeable
    assert y.flags.writeable

    opt.tell(x, y)

    # Verify that the arrays were marked read-only
    assert not x.flags.writeable
    assert not y.flags.writeable

    with pytest.raises(ValueError):
        x[0, 0] = 999.0
    with pytest.raises(ValueError):
        y[0] = 999.0
