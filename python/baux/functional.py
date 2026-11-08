"""Pure functional state transitions (`init`, `ask`, `tell`, `step`, `scan`, `pmap`) for BAUX."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from operator import index
import sys as _sys
from typing import Any

import numpy as np
from numpy.random import Generator

from . import _rust, pytree as _pytree, state as _state
from .bo import AskBatch, AskId, Telemetry, TellPredictionError, _obs
from .enn import enn_state as _enn
from .pytree import PyTreeSpec, pytree_bounds, ravel
from .state import (
    OptimizerState,
    _Cursor,
    _Op,
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

Key, Categorical, categorical = _rust.Key, _pytree.Categorical, _pytree.categorical
Affine, ENNState, Stiefel = _enn.Affine, _enn.ENNState, _enn.Stiefel
calibrate, stein = _enn.calibrate, _enn.stein
append, enn_append, enn_fit, enn_init = (
    _enn.append,
    _enn.enn_append,
    _enn.enn_fit,
    _enn.enn_init,
)
enn_posterior, enn_sample, fit, posterior, sample = (
    _enn.enn_posterior,
    _enn.enn_sample,
    _enn.fit,
    _enn.posterior,
    _enn.sample,
)
digest, freeze, thaw = _state.digest, _state.freeze, _state.thaw


def key(source: Any = None) -> Any:
    """Return a splittable 128-bit `Key` from a state/space/array/bytes, or fresh OS entropy when `None`."""
    if hasattr(source, "key"):
        return source.key
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
    key: Any = None,
    seed: int | Generator | None = None,
    low: Any = -1.0,
    high: Any = 1.0,
) -> OptimizerState:
    """Initialize an immutable `OptimizerState` from bounds or a parameter PyTree."""
    cfg = (
        OptimizerConfig()
        if config is None
        else (config() if isinstance(config, Transform) else config)
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
        pending_ids=None if raw_pending is None else tuple(int(i) for i in raw_pending),
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=Telemetry(
            t.dt_fit, t.dt_gen, t.dt_sel, t.dt_tell, int(t.num_candidates)
        ),
        _log=(),
        _cursor=_Cursor(0, inner),
    )


def ask(state: OptimizerState, num_arms: int = 1) -> tuple[OptimizerState, AskBatch]:
    """Pure transition `(state, num_arms) -> (next_state, batch)`."""
    n = int(num_arms)
    if n <= 0:
        raise ValueError(f"num_arms must be > 0, got {n}")
    inner, id_map, in_place = state._checkout()
    step_seed = _foldseed(state.seed, state.epoch + 1)
    before = set(inner.pending_ids() or ())
    points = _readonly(inner.ask(n, seed=step_seed))
    after = inner.pending_ids() or ()
    new_raw = tuple(int(i) for i in after if i not in before)
    inv_map = {v: k for k, v in id_map.items()}
    if len(new_raw) == n:
        logical_ids = tuple(int(inv_map.get(i, i)) for i in new_raw)
        for lid, rid in zip(logical_ids, new_raw):
            id_map[lid] = rid
    else:
        base_id = (state.epoch + 1) << 32
        logical_ids = tuple(base_id + i for i in range(n))
    op = _Op("ask", (n, logical_ids), step_seed)
    next_state = _snapshot(state, inner, id_map, in_place, op)
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
    inner, id_map, in_place = state._checkout()
    step_seed = _foldseed(state.seed, state.epoch + 1)
    kind = "import" if imported else "tell"
    if imported:
        inner.import_observations(obs_x, obs_y, seed=step_seed, y_var=obs_yvar)
    else:
        inner.tell(obs_x, obs_y, seed=step_seed, y_var=obs_yvar)
    op = _Op(kind, (obs_x, obs_y, obs_yvar), step_seed)
    next_state = _snapshot(state, inner, id_map, in_place, op)
    if not inner.has_surrogate():
        return next_state, _readonly(y_in)
    try:
        mu = np.asarray(inner.posterior_mu(obs_x), dtype=float)
    except Exception as error:
        raise TellPredictionError(
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
    raw_ids = tuple(index(i.value) for i in identities)
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
    inner, id_map, in_place = state._checkout()
    step_seed = _foldseed(state.seed, state.epoch + 1)
    mapped = [id_map.get(i, i) for i in raw_ids]
    inner.complete_ids(mapped, y_arr, seed=step_seed, y_var=yvar_arr)
    op = _Op("complete", (raw_ids, y_arr, yvar_arr), step_seed)
    return _snapshot(state, inner, id_map, in_place, op)


def cancel(
    state: OptimizerState,
    ids: AskBatch | tuple[AskId, ...] | list[AskId],
) -> OptimizerState:
    """Pure transition retiring unevaluated IDs without recording a reward."""
    identities = ids.ids if isinstance(ids, AskBatch) else tuple(ids)
    if any(not isinstance(i, AskId) for i in identities):
        raise TypeError("cancel expects AskId values")
    raw_ids = tuple(index(i.value) for i in identities)
    inner, id_map, in_place = state._checkout()
    step_seed = _foldseed(state.seed, state.epoch + 1)
    mapped = [id_map.get(i, i) for i in raw_ids]
    inner.cancel_ids(mapped, seed=step_seed)
    op = _Op("cancel", (raw_ids,), step_seed)
    return _snapshot(state, inner, id_map, in_place, op)


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
    inner, id_map, in_place = state._checkout()
    step_seed = _foldseed(state.seed, state.epoch + 1)
    inv_map = {v: k for k, v in id_map.items()}
    abandoned_raw = inner.restart_region(index(slot), center_ro, seed=step_seed)
    logical_abandoned = tuple(int(inv_map.get(int(i), int(i))) for i in abandoned_raw)
    op = _Op("restart", (index(slot), center_ro, logical_abandoned), step_seed)
    next_state = _snapshot(state, inner, id_map, in_place, op)
    return next_state, tuple(AskId(i) for i in logical_abandoned)


def posterior_mu(state: OptimizerState, x: Any) -> np.ndarray:
    """Pure query of fitted posterior means at `x` in natural units."""
    x_arr = _coercedpoints(state, x)
    if x_arr.ndim != 2 or x_arr.shape[1] != state.dim:
        raise ValueError(f"x must have shape (n, {state.dim}), got {x_arr.shape}")
    inner = state._readonlyview()
    return _readonly(inner.posterior_mu(x_arr))


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
    steps: int = 0,
    eval_fn: Callable[..., Any] | None = None,
    num_arms: int = 1,
) -> Any:
    """Fold `steps` pure optimization rounds over `state` and `key`, returning `(final_state, next_key, traces)`."""
    n_steps = int(steps)
    if n_steps < 0:
        raise ValueError(f"steps must be >= 0, got {n_steps}")
    if eval_fn is None and callable(key):
        cur, history = state, []
        for _ in range(n_steps):
            cur, y = step(cur, key, num_arms=num_arms, use_pytree=cur.spec is not None)
            history.append(y.reshape(-1))
        return cur, _readonly(
            np.vstack(history) if history else np.empty((0, int(num_arms)), dtype=float)
        )
    if eval_fn is None:
        raise ValueError("eval_fn must be provided when key is a PRNG key")
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
    root_key = state.key if key is None else (key if isinstance(key, Key) else Key(key))
    subkeys = split(root_key, n_branches)
    with ThreadPoolExecutor(max_workers=n_branches) as pool:
        return [
            f.result()
            for f in [
                pool.submit(branch_fn, i, state.fork(bk), bk)
                for i, bk in enumerate(subkeys)
            ]
        ]


if _pkg := _sys.modules.get(__package__ or "baux"):
    setattr(_pkg, "pmap", pmap)


__all__ = [
    "Affine",
    "Categorical",
    "ENNState",
    "Key",
    "OptimizerState",
    "Stiefel",
    "append",
    "ask",
    "calibrate",
    "cancel",
    "categorical",
    "complete",
    "digest",
    "enn_append",
    "enn_fit",
    "enn_init",
    "enn_posterior",
    "enn_sample",
    "fit",
    "fold",
    "freeze",
    "import_observations",
    "init",
    "key",
    "pmap",
    "posterior",
    "posterior_mu",
    "restart_region",
    "sample",
    "scan",
    "split",
    "stein",
    "step",
    "tell",
    "thaw",
]
