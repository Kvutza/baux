from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator

from . import _rust
from .turbo.config import OptimizerConfig
from .turbo.config.encode import encode, enn_k, is_lhd, supports
from .turbo.config.surrogate import ENNSurrogate, GaussianProcess, NoSurrogate
from .turbo.optimizer_types import Telemetry, _Obs, _Plan, _View


def _obs(x: np.ndarray, y: np.ndarray, yvar: np.ndarray | None, d: int) -> _Obs:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(f"x must have shape (n, {d}), got {x.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must have shape ({x.shape[0]}, m), got {y.shape}")
    if yvar is not None:
        yvar = np.asarray(yvar, dtype=float)
        if yvar.ndim == 1:
            yvar = yvar.reshape(-1, 1)
        if yvar.shape != y.shape:
            raise ValueError(f"y_var must have shape {y.shape}, got {yvar.shape}")
    return _Obs(x, y, yvar)


def _tellres(inner: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the fitted mean for a surrogate and raw observations otherwise."""
    if not inner.has_surrogate():
        return y
    mu = np.asarray(inner.posterior_mu(x), dtype=float)
    if y.ndim == 1 and mu.ndim == 2 and mu.shape[1] == 1:
        return mu.reshape(-1)
    return mu


def _plan(cfg: OptimizerConfig) -> _Plan:
    if is_lhd(cfg):
        return _Plan("lhd")
    if isinstance(cfg.surrogate, ENNSurrogate):
        return _Plan("enn", enn_k(cfg))
    if isinstance(cfg.surrogate, NoSurrogate):
        return _Plan("zero")
    if isinstance(cfg.surrogate, GaussianProcess):
        from .turbo.gp import Surrogate

        return _Plan("gp", gp=Surrogate())
    raise ValueError(f"unsupported surrogate: {type(cfg.surrogate)!r}")


class Optimizer:
    """Optimize a bounded search space through Rust-owned ask/tell state.

    ``ask`` returns parameter rows for objective evaluation. ``tell`` records
    objective values and optional observation variances. Objective evaluation
    belongs to the caller; constructing this object does not select a Python
    array framework or imply that posterior computation runs on an accelerator.
    """

    def __init__(self, d: int, rng: Generator, inner: Any) -> None:
        self._d = d
        self._rng = rng
        self._inner = inner

    @property
    def _xobs(self) -> _View:
        x = self._inner.x_obs()
        return _View(np.empty((0, self._d)) if x is None else x)

    @property
    def _yobs(self) -> _View:
        y = self._inner.y_obs()
        return _View(np.empty((0, 1)) if y is None else y)

    @property
    def region_count(self) -> int:
        return int(self._inner.region_count())

    @property
    def tr_length(self) -> float:
        return float(self._inner.tr_length())

    @property
    def init_progress(self) -> tuple[int, int] | None:
        return self._inner.init_progress()

    def telemetry(self) -> Telemetry:
        t = self._inner.telemetry()
        return Telemetry(
            dt_fit=t.dt_fit,
            dt_gen=t.dt_gen,
            dt_sel=t.dt_sel,
            dt_tell=t.dt_tell,
            num_candidates=int(t.num_candidates),
        )

    def ask(self, num_arms: int) -> np.ndarray:
        """Return `(num_arms, d)` candidates inside the configured bounds."""
        n = int(num_arms)
        if n <= 0:
            raise ValueError(f"num_arms must be > 0, got {n}")
        return np.asarray(self._inner.ask(n), dtype=float)

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        """Add an observation batch and return its 2D objective array."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr.flags.writeable = False
        y_arr.flags.writeable = False

        y_var_arr = None
        if y_var is not None:
            y_var_arr = np.asarray(y_var, dtype=float)
            y_var_arr.flags.writeable = False

        obs = _obs(x_arr, y_arr, y_var_arr, self._d)
        if obs.y.shape[0] == 0:
            return y_arr
        self._inner.tell(obs.x, obs.y, y_var=obs.yvar)
        return _tellres(self._inner, obs.x, y_arr)


def create_optimizer(
    *, bounds: np.ndarray, config: OptimizerConfig, rng: Generator
) -> Optimizer:
    """Build an optimizer from the Python config."""
    if not supports(config):
        raise ValueError(f"unsupported optimizer config: {config!r}")

    bounds = np.asarray(bounds, dtype=float)
    plan = _plan(config)
    n = config.init.num_init or 10
    seed = int(rng.integers(2**63 - 1))
    inner = _rust.create_optimizer(
        bounds,
        plan.kind,
        plan.k,
        n,
        4,
        seed,
        cfg=encode(config),
        gp=plan.gp,
    )
    return Optimizer(len(bounds), rng, inner)


# Preserve existing imports and isinstance checks during the public rename.
BayesOptimizer = Optimizer

__all__ = ["Optimizer", "Telemetry", "create_optimizer"]
