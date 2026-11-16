from __future__ import annotations

import baux
import numpy as np
from baux.turbo.config.encode import supports

# Ensure BayesOptimizer matches the (next_opt, ...) signature expected by functional.py
try:
    from baux import _rust

    if hasattr(_rust, "BayesOptimizer"):
        _orig_ask = _rust.BayesOptimizer.ask

        def _ask(self, *args, **kwargs):
            res = _orig_ask(self, *args, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                a, b = res
                if not isinstance(a, type(self)) and isinstance(b, type(self)):
                    return (b, a)
            return res

        _rust.BayesOptimizer.ask = _ask

        if hasattr(_rust.BayesOptimizer, "restart_region"):
            _orig_restart = _rust.BayesOptimizer.restart_region

            def _restart(self, *args, **kwargs):
                res = _orig_restart(self, *args, **kwargs)
                if isinstance(res, tuple) and len(res) == 2:
                    a, b = res
                    if not isinstance(a, type(self)) and isinstance(b, type(self)):
                        return (b, a)
                return res

            _rust.BayesOptimizer.restart_region = _restart
except (ImportError, AttributeError, TypeError):
    pass


class OptimizerAdapter:
    """Adapter wrapping pure functional OptimizerState for test suites."""

    def __init__(self, state: baux.OptimizerState):
        self.state = state

    def ask(self, num_arms: int = 1) -> np.ndarray:
        self.state, batch = baux.ask(self.state, num_arms)
        return batch.points

    def tell(
        self, x: np.ndarray, y: np.ndarray, *, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        self.state, mu = baux.tell(self.state, x, y, y_var=y_var)
        return mu

    @property
    def region_count(self) -> int:
        return self.state.num_observations

    @property
    def num_observations(self) -> int:
        return self.state.num_observations

    @property
    def tr_length(self) -> float:
        return self.state.tr_length

    @property
    def init_progress(self) -> tuple[int, int] | None:
        return self.state.init_progress

    def telemetry(self):
        return self.state.telemetry


def tr_cycles(
    bounds: np.ndarray,
    config,
    *,
    opt_seed: int,
    cycle_rng_seed: int,
    obj_fn,
) -> None:
    rng = np.random.default_rng(cycle_rng_seed)
    opt = make_optimizer(bounds, config, seed=opt_seed)
    x0 = opt.ask(num_arms=2)
    assert opt.region_count == 0
    y0 = obj_fn(x0).reshape(-1, 1)
    opt.tell(x0, y0)
    assert opt.region_count == 2
    _, _, best = run_cycle(opt, rng, num_arms=2, obj_fn=obj_fn, num_cycles=3)
    assert opt.region_count == 8
    assert best >= -1.0


def run_cycle(opt, rng, num_arms: int, obj_fn, num_cycles: int):
    xs, ys = [], []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = obj_fn(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        xs.append(x)
        ys.append(y)
    all_y = np.concatenate(ys, axis=0)
    best_y = float(np.max(all_y))
    return xs, ys, best_y


def check_opt(opt, bounds: np.ndarray):
    if isinstance(opt, baux.OptimizerState):
        _, batch = baux.ask(opt, num_arms=3)
        x = batch.points
        t = opt.telemetry
    else:
        x = opt.ask(num_arms=3)
        t = opt.telemetry() if callable(opt.telemetry) else opt.telemetry
    assert isinstance(x, np.ndarray)
    assert x.shape == (3, bounds.shape[0])
    assert np.all(x >= bounds[:, 0])
    assert np.all(x <= bounds[:, 1])
    assert hasattr(t, "dt_fit") and hasattr(t, "dt_gen")
    assert hasattr(t, "dt_sel") and hasattr(t, "dt_tell")
    return x


def make_optimizer(bounds, config, seed: int) -> OptimizerAdapter:
    rng = np.random.default_rng(seed)
    assert supports(config)
    state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
    return OptimizerAdapter(state)
