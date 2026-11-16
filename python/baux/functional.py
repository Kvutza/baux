"""Pure functional state transitions (`init`, `ask`, `tell`, `step`, `scan`, `pmap`) for baux."""

from __future__ import annotations

import sys as _sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from operator import index
from typing import Any

import numpy as np
from numpy.random import Generator

from . import pytree as _pytree
from . import state as _state
from ._lazy import module_attr
from .bo import AskBatch, AskId, Telemetry, _obs
from .pytree import PyTreeSpec, pytree_bounds, ravel
from .state import (
    OptimizerState,
    _foldseed,
    _makeinner,
    _readonly,
    _resolveseed,
    _snapshot,
)
from .transforms import Transform
from .turbo.config import OptimizerConfig
from .turbo.config.encode import supports
from .turbo.config.surrogate import ENNSurrogate
from .turbo.config.trust_region import RegionalSearchPolicy

Affine, Categorical, Stiefel, categorical = (
    _pytree.Affine,
    _pytree.Categorical,
    _pytree.Stiefel,
    _pytree.categorical,
)
digest, freeze, thaw = _state.digest, _state.freeze, _state.thaw

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ENNState": (".enn.enn_state", "ENNState"),
    "Key": ("._rust", "Key"),
    "append": (".enn.enn_state", "append"),
    "calibrate": (".enn.enn_state", "calibrate"),
    "enn_init": (".enn.enn_state", "enn_init"),
    "ensure_sync": (".enn.enn_state", "ensure_sync"),
    "fit": (".enn.enn_state", "fit"),
    "persist_index": (".enn.enn_state", "persist_index"),
    "posterior": (".enn.enn_state", "posterior"),
    "sample": (".enn.enn_state", "sample"),
    "stein": (".enn.enn_state", "stein"),
    "with_geometry": (".enn.enn_state", "with_geometry"),
}


def __getattr__(name: str) -> Any:
    return module_attr(name, globals())


def key(source: Any = None) -> Any:
    """Return a splittable 128-bit `Key` from a state/space/array/bytes, or fresh OS entropy when `None`."""
    if hasattr(source, "key"):
        return source.key
    from ._rust import Key

    if isinstance(source, np.ndarray):
        return Key.from_bytes(np.ascontiguousarray(source).tobytes())
    return Key(source)


def split(k: Any, n: int = 2) -> tuple[Any, ...]:
    """Deterministically split `k` into `n` independent subkeys."""
    count = int(n)
    if count == 2:
        a, b = k.split()
        return (a, b)
    return tuple(k.split_n(count))


def fold(k: Any, tag: int) -> Any:
    """Deterministically fold an integer `tag` into `k`."""
    return k.fold(int(tag) & ((1 << 64) - 1))


def _parsedomain(
    domain: Any, low: Any, high: Any
) -> tuple[PyTreeSpec | None, np.ndarray]:
    if isinstance(domain, PyTreeSpec):
        return domain, domain.bounds
    if isinstance(domain, np.ndarray) or (
        isinstance(domain, (list, tuple))
        and len(domain) > 0
        and isinstance(domain[0], (list, tuple, np.ndarray))
        and len(domain[0]) == 2
        and not isinstance(domain[0], dict)
    ):
        bounds = _readonly(domain)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")
        return None, bounds
    spec = pytree_bounds(domain, low=low, high=high)
    return spec, spec.bounds


def init(
    domain: np.ndarray | PyTreeSpec | Any,
    config: OptimizerConfig | Transform | None = None,
    *,
    pipe: OptimizerConfig | Transform | None = None,
    key: Any = None,
    seed: int | Generator | None = None,
    low: Any = -1.0,
    high: Any = 1.0,
) -> OptimizerState:
    """Initialize an immutable `OptimizerState` from bounds or a parameter PyTree."""
    if config is not None and pipe is not None:
        raise ValueError("pass either `config` or `pipe`, not both")
    raw_cfg = pipe if config is None else config
    cfg = (
        OptimizerConfig()
        if raw_cfg is None
        else (raw_cfg() if isinstance(raw_cfg, Transform) else raw_cfg)
    )
    if not supports(cfg):
        raise ValueError(f"unsupported optimizer config: {cfg!r}")
    spec, bounds = _parsedomain(domain, low, high)
    if spec is not None and spec.tied_dims:
        from dataclasses import replace as _replace

        if cfg.candidates.tied_dims is None:
            cfg = _replace(
                cfg, candidates=_replace(cfg.candidates, tied_dims=spec.tied_dims)
            )
        if isinstance(cfg.surrogate, ENNSurrogate) and cfg.surrogate.tied_dims is None:
            cfg = _replace(
                cfg, surrogate=_replace(cfg.surrogate, tied_dims=spec.tied_dims)
            )
    default_bytes = (
        spec.key.seed().to_bytes(8, "little")
        if spec is not None
        else np.ascontiguousarray(bounds, dtype=float).tobytes()
    ) + repr(cfg).encode("utf-8")
    base_seed = _resolveseed(seed, key, default_bytes)
    inner = _makeinner(bounds, cfg, base_seed)
    t = inner.telemetry()
    raw_pending = inner.pending_ids()
    return OptimizerState(
        dim=int(bounds.shape[0]),
        bounds=bounds,
        config=cfg,
        seed=base_seed,
        epoch=0,
        spec=spec,
        identified=isinstance(cfg.trust_region, RegionalSearchPolicy),
        num_observations=int(inner.num_observations()),
        num_regions=int(inner.num_regions()),
        tr_length=float(inner.tr_length()),
        init_progress=inner.init_progress(),
        pending_ids=None if not raw_pending else tuple(int(i) for i in raw_pending),
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=Telemetry(
            t.dt_fit, t.dt_gen, t.dt_sel, t.dt_tell, int(t.num_candidates)
        ),
        _opt=inner,
    )


def ask(
    state: OptimizerState,
    num_arms: int = 1,
) -> tuple[OptimizerState, AskBatch]:
    """Pure transition `(state, num_arms) -> (next_state, batch)`."""
    arms = int(num_arms)
    if arms <= 0:
        raise ValueError(f"num_arms must be > 0, got {arms}")
    step_seed = _foldseed(state.seed, state.epoch + 1)
    before = set(state._opt.pending_ids() or ())
    next_opt, raw = state._opt.ask(arms, seed=step_seed)
    points = _readonly(raw)
    after = next_opt.pending_ids() or ()
    new_raw = tuple(int(i) for i in after if i not in before)
    if len(new_raw) == arms:
        logical_ids = tuple(int(i) for i in new_raw)
    else:
        base_id = (state.epoch + 1) << 32
        logical_ids = tuple(base_id + i for i in range(arms))
    next_state = _snapshot(state, next_opt)
    batch = AskBatch(
        ids=tuple(AskId(i) for i in logical_ids),
        points=points,
        spec=state.spec,
        weights=next_state.metric_weights,
    )
    return next_state, batch


def _coercedpoints(state: OptimizerState, x: Any) -> np.ndarray:
    if isinstance(x, AskBatch):
        return x.points
    if state.spec is not None and not isinstance(x, np.ndarray):
        arr = ravel(x, state.spec)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _readonly(arr)
    arr = _readonly(x)
    if arr.ndim == 1 and arr.shape[0] == state.dim:
        arr = arr.reshape(1, state.dim)
    return arr


def _submitpure(
    state: OptimizerState,
    x: Any,
    y: Any,
    y_var: Any,
    *,
    imported: bool,
) -> tuple[OptimizerState, np.ndarray]:
    x_arr = _coercedpoints(state, x)
    y_in = np.asarray(y, dtype=float)
    yvar_in = None if y_var is None else _readonly(y_var)
    obs = _obs(x_arr, _readonly(y_in), yvar_in, state.dim)
    if obs.y.shape[0] == 0:
        return state, _readonly(y_in)
    obs_x = _readonly(obs.x)
    obs_y = _readonly(obs.y)
    obs_yvar = None if obs.yvar is None else _readonly(obs.yvar)
    step_seed = _foldseed(state.seed, state.epoch + 1)
    if imported:
        next_opt = state._opt.import_observations(
            obs_x, obs_y, seed=step_seed, y_var=obs_yvar
        )
    else:
        next_opt = state._opt.tell(obs_x, obs_y, seed=step_seed, y_var=obs_yvar)
    next_state = _snapshot(state, next_opt)
    if not next_opt.has_surrogate():
        return next_state, _readonly(y_in)
    try:
        mu = np.asarray(next_opt.posterior_mu(obs_x), dtype=float)
    except Exception as error:
        raise RuntimeError(
            "observations accepted, but prediction retrieval failed; "
            "retry posterior_mu(state, x), not the submission"
        ) from error
    if y_in.ndim == 1 and mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    return next_state, _readonly(mu)


def tell(
    state: OptimizerState,
    x: Any,
    y: Any,
    *,
    y_var: Any = None,
) -> tuple[OptimizerState, np.ndarray]:
    """Pure transition `(state, x, y) -> (next_state, fitted_mu)` leaving `state` untouched."""
    return _submitpure(state, x, y, y_var, imported=False)


def import_observations(
    state: OptimizerState,
    x: Any,
    y: Any,
    *,
    y_var: Any = None,
) -> tuple[OptimizerState, np.ndarray]:
    """Pure transition adding external observations without completing a pending batch."""
    return _submitpure(state, x, y, y_var, imported=True)


def complete(
    state: OptimizerState,
    ids: AskBatch | tuple[AskId, ...] | list[AskId],
    y: Any,
    *,
    y_var: Any = None,
) -> OptimizerState:
    """Pure transition completing a subset of pending regional evaluation IDs."""
    identities = ids.ids if isinstance(ids, AskBatch) else tuple(ids)
    if any(not isinstance(i, AskId) for i in identities):
        raise TypeError("complete expects AskId values")
    raw_ids = [index(i.value) for i in identities]
    y_arr = _readonly(y)
    if y_arr.ndim == 1:
        y_arr = _readonly(y_arr.reshape(-1, 1))
    if y_arr.shape != (len(raw_ids), 1):
        raise ValueError(f"y must have shape ({len(raw_ids)}, 1), got {y_arr.shape}")
    yvar_arr = None if y_var is None else _readonly(y_var)
    if yvar_arr is not None:
        if yvar_arr.ndim == 1:
            yvar_arr = _readonly(yvar_arr.reshape(-1, 1))
        if yvar_arr.shape != y_arr.shape:
            raise ValueError(
                f"y_var must have shape {y_arr.shape}, got {yvar_arr.shape}"
            )
    step_seed = _foldseed(state.seed, state.epoch + 1)
    next_opt = state._opt.complete_ids(raw_ids, y_arr, seed=step_seed, y_var=yvar_arr)
    return _snapshot(state, next_opt)


def cancel(
    state: OptimizerState,
    ids: AskBatch | tuple[AskId, ...] | list[AskId],
) -> OptimizerState:
    """Pure transition retiring unevaluated IDs without recording a reward."""
    identities = ids.ids if isinstance(ids, AskBatch) else tuple(ids)
    if any(not isinstance(i, AskId) for i in identities):
        raise TypeError("cancel expects AskId values")
    raw_ids = [index(i.value) for i in identities]
    step_seed = _foldseed(state.seed, state.epoch + 1)
    next_opt = state._opt.cancel_ids(raw_ids, seed=step_seed)
    return _snapshot(state, next_opt)


def restart_region(
    state: OptimizerState,
    slot: int,
    center: Any,
) -> tuple[OptimizerState, tuple[AskId, ...]]:
    """Pure transition restarting a regional slot at `center`, returning `(next_state, abandoned_ids)`."""
    if not state.identified:
        raise RuntimeError("region restart requires a multi-region optimizer")
    center_arr = _coercedpoints(state, center).reshape(-1)
    if center_arr.shape != (state.dim,):
        raise ValueError(
            f"center must have shape ({state.dim},), got {center_arr.shape}"
        )
    center_ro = _readonly(center_arr)
    step_seed = _foldseed(state.seed, state.epoch + 1)
    next_opt, abandoned_raw = state._opt.restart_region(
        index(slot), center_ro, seed=step_seed
    )
    logical_abandoned = tuple(int(i) for i in abandoned_raw)
    next_state = _snapshot(state, next_opt)
    return next_state, tuple(AskId(i) for i in logical_abandoned)


def posterior_mu(state: OptimizerState, x: Any) -> np.ndarray:
    """Pure query of fitted posterior means at `x` in natural units."""
    x_arr = _coercedpoints(state, x)
    if x_arr.ndim != 2 or x_arr.shape[1] != state.dim:
        raise ValueError(f"x must have shape (n, {state.dim}), got {x_arr.shape}")
    return _readonly(state._opt.posterior_mu(x_arr))


def step(
    state: OptimizerState,
    objective: Callable[[Any], Any],
    *,
    num_arms: int = 1,
    use_pytree: bool = False,
) -> tuple[OptimizerState, np.ndarray]:
    """Execute one pure `(ask -> objective -> tell)` step."""
    s1, batch = ask(state, num_arms)
    arg = (
        (batch.stacked if num_arms > 1 else batch.params[0])
        if (use_pytree and state.spec is not None)
        else batch.points
    )
    y = np.asarray(objective(arg), dtype=float)
    s2, _ = tell(s1, batch, y)
    return s2, _readonly(y)


class _StepTrace(dict[str, Any]):
    def __init__(self, state: OptimizerState, step: int, best_y: float | None) -> None:
        super().__init__(step=step, best_y=best_y)
        self._state = state

    def __getitem__(self, key: Any) -> Any:
        if key == "digest" and "digest" not in self:
            self["digest"] = self._state.digest()
        return super().__getitem__(key)

    def get(self, key: object, default: Any = None, /) -> Any:
        if key == "digest" and "digest" not in self:
            self["digest"] = self._state.digest()
        return super().get(key, default)


def scan(
    state: OptimizerState,
    key: Any,
    steps: int,
    eval_fn: Callable[..., Any],
    num_arms: int = 1,
) -> tuple[OptimizerState, Any, list[dict[str, Any]]]:
    """Fold `steps` pure optimization rounds over `state` and `key`, returning `(final_state, next_key, traces)`."""
    n_steps = int(steps)
    if n_steps < 0:
        raise ValueError(f"steps must be >= 0, got {n_steps}")
    if key is None:
        cur_key = state.key
    else:
        from ._rust import Key

        cur_key = key if isinstance(key, Key) else Key(key)
    cur = state.fork(cur_key)
    best = cur.best_y
    traces: list[dict[str, Any]] = []
    for t in range(n_steps):
        step_key, cur_key = split(cur_key, 2)
        s1, batch = ask(cur.fork(step_key), num_arms)
        try:
            y_val = eval_fn(batch.points)
        except TypeError:
            y_val = eval_fn(t, batch.points)
        y_arr = np.asarray(y_val, dtype=float)
        cur, _ = tell(s1, batch, y_arr)
        y_col = y_arr.reshape(len(batch), -1)[:, 0]
        if y_col.size > 0:
            y_max = float(np.max(y_col))
            best = y_max if best is None or y_max > best else best
        traces.append(_StepTrace(cur, t, best))
    return cur, cur_key, traces


def pmap(
    state: OptimizerState,
    key: Any,
    branches: int,
    branch_fn: Callable[[int, OptimizerState, Any], Any],
) -> list[Any]:
    """Fork `state` and split `key` across `branches`, executing `branch_fn` in parallel across threads."""
    n_branches = int(branches)
    if n_branches < 0:
        raise ValueError(f"branches must be >= 0, got {n_branches}")
    if n_branches == 0:
        return []
    if key is None:
        root_key = state.key
    else:
        from ._rust import Key

        root_key = key if isinstance(key, Key) else Key(key)
    subkeys = split(root_key, n_branches)
    if _sys.platform == "emscripten" or n_branches == 1:
        return [branch_fn(i, state.fork(bk), bk) for i, bk in enumerate(subkeys)]
    with ThreadPoolExecutor(max_workers=n_branches) as pool:
        return [
            f.result()
            for f in [
                pool.submit(branch_fn, i, state.fork(bk), bk)
                for i, bk in enumerate(subkeys)
            ]
        ]


if _pkg := _sys.modules.get(__package__ or "baux"):
    _pkg.__dict__["pmap"] = pmap


__all__ = [
    "Affine",
    "Categorical",
    "OptimizerState",
    "Stiefel",
    "ask",
    "cancel",
    "categorical",
    "complete",
    "digest",
    "fold",
    "freeze",
    "import_observations",
    "init",
    "key",
    "pmap",
    "posterior_mu",
    "restart_region",
    "scan",
    "split",
    "step",
    "tell",
    "thaw",
]
