"""Immutable, forkable `OptimizerState` value token and COW cursor management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .bo import Optimizer, Telemetry, _plan
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


def _resolveseed(
    seed: Any = None, key: Any = None, default_bytes: bytes | None = None
) -> int:
    src = key if key is not None else seed
    if src is None:
        from . import _rust

        k = (
            _rust.Key()
            if default_bytes is None
            else _rust.Key.from_bytes(default_bytes)
        )
        return int(k.seed()) & ((1 << 63) - 1)
    if isinstance(src, (bytes, bytearray)):
        from . import _rust

        return int(_rust.Key.from_bytes(bytes(src)).seed()) & ((1 << 63) - 1)
    if isinstance(src, Generator):
        return int(src.integers(2**63 - 1))
    if hasattr(src, "key"):
        src = src.key
    if hasattr(src, "seed") and callable(src.seed):
        return int(src.seed()) & ((1 << 63) - 1)
    return int(src) & ((1 << 63) - 1)


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
        for orig, cur in zip(orig_ids, new_ids, strict=True):
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
        for orig, cur in zip(orig_abandoned, abandoned, strict=True):
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

    @property
    def _opt(self) -> Optimizer:
        return Optimizer(
            self.dim,
            np.random.default_rng(self.seed),
            self._readonlyview(),
            identified_asks=self.identified,
        )

    def _checkout(self) -> tuple[Any, dict[int, int], bool]:
        is_gp = isinstance(self.config.surrogate, (GaussianProcess, ExternalSurrogate))
        if (
            self._cursor.epoch == self.epoch
            and not is_gp
            and hasattr(self._cursor.inner, "fork")
        ):
            try:
                return self._cursor.inner.fork(), dict(self._cursor.id_map), False
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
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
    def x_obs(self) -> np.ndarray:
        """Read-only `(n, dim)` matrix of observed parameter vectors."""
        x = self._readonlyview().x_obs()
        return _readonly(np.empty((0, self.dim)) if x is None else x)

    @property
    def y_obs(self) -> np.ndarray:
        """Read-only `(n, m)` matrix of observed objective values."""
        y = self._readonlyview().y_obs()
        return _readonly(np.empty((0, 1)) if y is None else y)

    @property
    def incumbent_x(self) -> np.ndarray | None:
        """Read-only incumbent parameter vector, or `None` if no observations exist."""
        inc = self._readonlyview().incumbent_x()
        return None if inc is None else _readonly(inc)

    @property
    def lengthscales(self) -> np.ndarray | None:
        """Read-only per-dimension surrogate lengthscales, if supported."""
        view = self._readonlyview()
        if not hasattr(view, "lengthscales"):
            return None
        ls = view.lengthscales()
        return None if ls is None else _readonly(ls).reshape(-1)

    @property
    def metric_weights(self) -> np.ndarray | None:
        """Read-only inverse lengthscale weights used for metric-aware decoding."""
        ls = self.lengthscales
        return None if ls is None else _readonly(1.0 / np.maximum(ls, 1e-12))

    @property
    def incumbent_params(self) -> Any:
        """Incumbent parameters decoded through `spec`, or `incumbent_x` if unspec'd."""
        inc = self.incumbent_x
        if inc is None or self.spec is None:
            return inc
        return self.spec.unflatten(inc.reshape(-1), weights=self.metric_weights)

    @property
    def best_y(self) -> float | None:
        """Best observed primary objective value, or `None` if empty."""
        y = self.y_obs
        return None if y.size == 0 else float(np.max(y[:, 0]))

    def digest(self) -> str:
        """Return the 64-character hexadecimal BLAKE3 state digest."""
        return str(self._readonlyview().digest_hex())

    @property
    def key(self) -> Any:
        """Deterministic PRNG `Key` derived from this state's BLAKE3 digest."""
        from . import _rust

        return _rust.Key.from_bytes(bytes.fromhex(self.digest()))

    def fork(self, key: Any = None, *, seed: Any = None) -> OptimizerState:
        """Return an independent copy of this state with an optional new `key` or `seed`."""
        from dataclasses import replace

        inner, id_map, in_place = self._checkout()
        if in_place and hasattr(inner, "fork"):
            inner = inner.fork()
        src = key if key is not None else seed
        return replace(
            self,
            seed=self.seed if src is None else _resolveseed(src),
            _cursor=_Cursor(self.epoch, inner, id_map),
        )


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
        num_observations=int(inner.num_observations()),
        num_regions=int(inner.num_regions()),
        tr_length=float(inner.tr_length()),
        init_progress=inner.init_progress(),
        pending_ids=pending,
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=telem,
        _log=(*prev._log, op) if in_place else (),
        _cursor=cursor,
    )


def digest(state: OptimizerState) -> str:
    """Return the 64-character lowercase hexadecimal BLAKE3 digest of `state`."""
    return str(state._readonlyview().digest_hex())


def freeze(state: OptimizerState) -> bytes:
    """Serialize `state` into a portable binary checkpoint."""
    return bytes(state._readonlyview().freeze())


def thaw(state: OptimizerState, data: bytes) -> OptimizerState:
    """Restore an `OptimizerState` from a binary checkpoint without mutating `state`."""
    inner = state._readonlyview().fork()
    inner.thaw(bytes(data))
    t = inner.telemetry()
    telem = Telemetry(
        dt_fit=t.dt_fit,
        dt_gen=t.dt_gen,
        dt_sel=t.dt_sel,
        dt_tell=t.dt_tell,
        num_candidates=int(t.num_candidates),
    )
    raw_pending = inner.pending_ids()
    pending = None if raw_pending is None else tuple(int(i) for i in raw_pending)
    next_epoch = state.epoch + 1
    cursor = _Cursor(next_epoch, inner, {})
    return OptimizerState(
        dim=state.dim,
        bounds=state.bounds,
        config=state.config,
        seed=state.seed,
        epoch=next_epoch,
        spec=state.spec,
        identified=state.identified,
        num_observations=int(inner.num_observations()),
        num_regions=int(inner.num_regions()),
        tr_length=float(inner.tr_length()),
        init_progress=inner.init_progress(),
        pending_ids=pending,
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=telem,
        _log=(),
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
    "digest",
    "freeze",
    "thaw",
]
