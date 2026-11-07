"""Immutable, forkable `OptimizerState` value token and COW cursor management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .bo import Telemetry, _plan
from .pytree import PyTreeSpec
from .turbo.config import OptimizerConfig
from .turbo.config.encode import encode
from .turbo.config.surrogate import ExternalSurrogate, GaussianProcess
from .turbo.config.trust_region import RegionalSearchPolicy

_MASK64 = (1 << 64) - 1


def _foldseed(seed: int, salt: int) -> int:
    z = (int(seed) ^ ((int(salt) + 0x9E3779B97F4A7C15) & _MASK64)) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return int((z ^ (z >> 31)) & ((1 << 63) - 1))


def _resolveseed(seed: int | Generator | None) -> int:
    if seed is None:
        return 0x5EED_BA08_2026
    if isinstance(seed, Generator):
        return int(seed.integers(2**63 - 1))
    return int(seed) & ((1 << 63) - 1)


def _readonly(arr: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class _Op:
    kind: str
    args: tuple[Any, ...]
    seed: int


class _Cursor:
    def __init__(
        self,
        epoch: int,
        inner: Any,
        id_map: dict[int, int] | None = None,
    ) -> None:
        self.epoch = epoch
        self.inner = inner
        self.id_map: dict[int, int] = {} if id_map is None else dict(id_map)


def _makeinner(bounds: np.ndarray, config: OptimizerConfig, seed: int) -> Any:
    from . import _rust

    plan = _plan(config)
    n = 10 if config.init.num_init is None else int(config.init.num_init)
    regions = (
        config.trust_region.count
        if isinstance(config.trust_region, RegionalSearchPolicy)
        else 1
    )
    provider: Callable[[], Any] | None = None
    if isinstance(config.trust_region, RegionalSearchPolicy):
        if isinstance(config.surrogate, ExternalSurrogate):
            provider = config.surrogate.factory
        elif isinstance(config.surrogate, GaussianProcess):
            from .turbo.gp import Surrogate

            provider = Surrogate
    gp_factory = (lambda slot, gen: provider()) if provider is not None else None
    return _rust.create_optimizer(
        bounds,
        plan.kind,
        plan.k,
        n,
        regions,
        seed,
        cfg=encode(config),
        gp=plan.gp,
        gp_factory=gp_factory,
        fit_steps=plan.fit_steps,
    )


def _applyop(inner: Any, op: _Op, id_map: dict[int, int]) -> None:
    if op.kind == "ask":
        num_arms, orig_ids = op.args
        before = set(inner.pending_ids() or ())
        inner.ask(num_arms, seed=op.seed)
        after = inner.pending_ids() or ()
        new_ids = [i for i in after if i not in before]
        for orig, cur in zip(orig_ids, new_ids):
            id_map[orig] = cur
    elif op.kind == "tell":
        x, y, yvar = op.args
        inner.tell(x, y, seed=op.seed, y_var=yvar)
    elif op.kind == "import":
        x, y, yvar = op.args
        inner.import_observations(x, y, seed=op.seed, y_var=yvar)
    elif op.kind == "complete":
        ids, y, yvar = op.args
        mapped = [id_map.get(i, i) for i in ids]
        inner.complete_ids(mapped, y, seed=op.seed, y_var=yvar)
    elif op.kind == "cancel":
        (ids,) = op.args
        mapped = [id_map.get(i, i) for i in ids]
        inner.cancel_ids(mapped, seed=op.seed)
    elif op.kind == "restart":
        slot, center, orig_abandoned = op.args
        abandoned = inner.restart_region(slot, center, seed=op.seed)
        for orig, cur in zip(orig_abandoned, abandoned):
            id_map[orig] = cur


@dataclass(frozen=True)
class OptimizerState:
    """Immutable, forkable value token representing the full state of a BAUX optimizer."""

    dim: int
    bounds: np.ndarray
    config: OptimizerConfig
    seed: int
    epoch: int
    spec: PyTreeSpec | None
    identified: bool
    num_observations: int
    num_regions: int
    tr_length: float
    init_progress: tuple[int, int] | None
    pending_ids: tuple[int, ...] | None
    ingestion_failure: str | None
    telemetry: Telemetry
    _log: tuple[_Op, ...]
    _cursor: _Cursor

    def _checkout(self) -> tuple[Any, dict[int, int], bool]:
        if self._cursor.epoch == self.epoch:
            return self._cursor.inner, self._cursor.id_map, True
        inner = _makeinner(self.bounds, self.config, self.seed)
        id_map: dict[int, int] = {}
        for op in self._log:
            _applyop(inner, op, id_map)
        return inner, id_map, False

    def _readonlyview(self) -> Any:
        if self._cursor.epoch == self.epoch:
            return self._cursor.inner
        inner, id_map, _ = self._checkout()
        object.__setattr__(self, "_cursor", _Cursor(self.epoch, inner, id_map))
        return inner

    @property
    def region_count(self) -> int:
        return self.num_observations

    @property
    def x_obs(self) -> np.ndarray:
        x = self._readonlyview().x_obs()
        return _readonly(np.empty((0, self.dim)) if x is None else x)

    @property
    def y_obs(self) -> np.ndarray:
        y = self._readonlyview().y_obs()
        return _readonly(np.empty((0, 1)) if y is None else y)

    @property
    def incumbent_x(self) -> np.ndarray | None:
        inc = self._readonlyview().incumbent_x()
        return None if inc is None else _readonly(inc)

    @property
    def incumbent_params(self) -> Any:
        inc = self.incumbent_x
        if inc is None or self.spec is None:
            return inc
        return self.spec.unflatten(inc.reshape(-1))


def _snapshot(
    prev: OptimizerState,
    inner: Any,
    id_map: dict[int, int],
    in_place: bool,
    op: _Op,
) -> OptimizerState:
    t = inner.telemetry()
    telem = Telemetry(
        dt_fit=t.dt_fit,
        dt_gen=t.dt_gen,
        dt_sel=t.dt_sel,
        dt_tell=t.dt_tell,
        num_candidates=int(t.num_candidates),
    )
    inv_map = {v: k for k, v in id_map.items()}
    raw_pending = inner.pending_ids()
    pending = (
        None
        if raw_pending is None
        else tuple(int(inv_map.get(int(i), int(i))) for i in raw_pending)
    )
    next_epoch = prev.epoch + 1
    if in_place:
        prev._cursor.epoch = next_epoch
        prev._cursor.id_map = id_map
        cursor = prev._cursor
    else:
        cursor = _Cursor(next_epoch, inner, id_map)
    return OptimizerState(
        dim=prev.dim,
        bounds=prev.bounds,
        config=prev.config,
        seed=prev.seed,
        epoch=next_epoch,
        spec=prev.spec,
        identified=prev.identified,
        num_observations=int(inner.region_count()),
        num_regions=int(inner.num_regions()),
        tr_length=float(inner.tr_length()),
        init_progress=inner.init_progress(),
        pending_ids=pending,
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=telem,
        _log=(*prev._log, op),
        _cursor=cursor,
    )


__all__ = [
    "OptimizerState",
    "_Cursor",
    "_Op",
    "_foldseed",
    "_makeinner",
    "_readonly",
    "_resolveseed",
    "_snapshot",
]
