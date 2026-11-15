"""Immutable, forkable `OptimizerState` value token backed by persistent Rust optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
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
    kind: str = ""
    args: tuple[Any, ...] = ()
    seed: int = 0


class _Cursor:
    def __init__(
        self,
        epoch: int = 0,
        inner: Any = None,
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


@dataclass(frozen=True)
class OptimizerState:
    """Immutable, forkable value token representing the full state of a baux optimizer."""

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
    _opt: Any
    _log: tuple[object, ...] = ()

    @property
    def _cursor(self) -> _Cursor:
        return _Cursor(self.epoch, self._opt)

    def _readonlyview(self) -> Any:
        return self._opt

    def _checkout(self) -> tuple[Any, dict[int, int], bool]:
        return self._opt, {}, False

    @property
    def x_obs(self) -> np.ndarray:
        """Read-only `(n, dim)` matrix of observed parameter vectors."""
        x = self._opt.x_obs()
        return _readonly(np.empty((0, self.dim)) if x is None else x)

    @property
    def y_obs(self) -> np.ndarray:
        """Read-only `(n, m)` matrix of observed objective values."""
        y = self._opt.y_obs()
        return _readonly(np.empty((0, 1)) if y is None else y)

    @property
    def incumbent_x(self) -> np.ndarray | None:
        """Read-only incumbent parameter vector, or `None` if no observations exist."""
        inc = self._opt.incumbent_x()
        return None if inc is None else _readonly(inc)

    @property
    def lengthscales(self) -> np.ndarray | None:
        """Read-only per-dimension surrogate lengthscales, if supported."""
        if not hasattr(self._opt, "lengthscales"):
            return None
        ls = self._opt.lengthscales()
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
    def best_x(self) -> np.ndarray | None:
        """Read-only best observed parameter vector (alias for `incumbent_x`)."""
        return self.incumbent_x

    @property
    def best_params(self) -> Any:
        """Best observed parameters decoded through `spec` (alias for `incumbent_params`)."""
        return self.incumbent_params

    @property
    def best_y(self) -> float | None:
        """Best observed primary objective value, or `None` if empty."""
        y = self.y_obs
        return None if y.size == 0 else float(np.max(y[:, 0]))

    @property
    def incumbent_y(self) -> float | None:
        """Best observed primary objective value (alias for `best_y`)."""
        return self.best_y

    def digest(self) -> str:
        """Return the 64-character hexadecimal BLAKE3 state digest."""
        if hasattr(self._opt, "digest_host"):
            return str(
                self._opt.digest_host(
                    int(self.seed), int(self.epoch), bool(self.identified)
                )
            )
        return str(self._opt.digest_hex())

    @property
    def key(self) -> Any:
        """Deterministic PRNG `Key` derived from this state's BLAKE3 digest."""
        from . import _rust

        return _rust.Key.from_bytes(bytes.fromhex(self.digest()))

    def fork(self, key: Any = None, *, seed: Any = None) -> OptimizerState:
        """Return an independent copy of this state with an optional new `key` or `seed`."""
        src = key if key is not None else seed
        new_seed = self.seed if src is None else _resolveseed(src)
        new_opt = self._opt.fork() if hasattr(self._opt, "fork") else self._opt
        return replace(
            self,
            seed=new_seed,
            _opt=new_opt,
            _log=(),
        )

    def freeze(self) -> bytes:
        """Serialize this state into a portable binary checkpoint."""
        return freeze(self)

    def thaw(self, data: bytes) -> OptimizerState:
        """Restore an `OptimizerState` from a binary checkpoint using this state as a template."""
        return thaw(self, data)


def _snapshot(
    prev: OptimizerState,
    inner: Any,
    id_map: Any = None,
    in_place: bool = False,
    op: Any = None,
) -> OptimizerState:
    t = inner.telemetry()
    telem = Telemetry(
        dt_fit=t.dt_fit,
        dt_gen=t.dt_gen,
        dt_sel=t.dt_sel,
        dt_tell=t.dt_tell,
        num_candidates=int(t.num_candidates),
    )
    raw_pending = inner.pending_ids()
    pending = None if not raw_pending else tuple(int(i) for i in raw_pending)
    return OptimizerState(
        dim=prev.dim,
        bounds=prev.bounds,
        config=prev.config,
        seed=prev.seed,
        epoch=prev.epoch + 1,
        spec=prev.spec,
        identified=prev.identified,
        num_observations=int(inner.num_observations()),
        num_regions=int(inner.num_regions()),
        tr_length=float(inner.tr_length()),
        init_progress=inner.init_progress(),
        pending_ids=pending,
        ingestion_failure=getattr(inner, "ingestion_failure", None),
        telemetry=telem,
        _opt=inner,
        _log=(),
    )


def digest(state: OptimizerState) -> str:
    """Return the 64-character lowercase hexadecimal BLAKE3 digest of `state`."""
    return state.digest()


def freeze(state: OptimizerState) -> bytes:
    """Serialize `state` into a portable binary checkpoint."""
    view = state._opt
    if hasattr(view, "freeze_host"):
        pairs: list[tuple[int, int]] = []
        try:
            return bytes(
                view.freeze_host(
                    int(state.seed), int(state.epoch), bool(state.identified), pairs
                )
            )
        except TypeError:
            return bytes(
                view.freeze_host(
                    int(state.seed), int(state.epoch), bool(state.identified)
                )
            )
    return bytes(view.freeze())


def thaw(state: OptimizerState, data: bytes) -> OptimizerState:
    """Restore an `OptimizerState` from a binary checkpoint without mutating `state`."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("checkpoint data must be a bytes-like object")
    raw = bytes(data)
    if len(raw) == 0:
        raise ValueError("checkpoint data is empty")
    inner = state._opt
    if hasattr(inner, "thaw_host"):
        try:
            res = inner.thaw_host(raw)
        except TypeError:
            res = inner.thaw_host(
                raw,
                int(state.seed),
                int(state.epoch),
                bool(state.identified),
            )
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple):
            (seed, epoch, identified, _pairs), next_opt = res
        elif isinstance(res, tuple) and len(res) == 4:
            seed, epoch, identified, _pairs = res
            next_opt = inner
        else:
            raise ValueError("unexpected thaw_host result format")
    elif hasattr(inner, "thaw"):
        next_opt = inner.thaw(raw)
        seed, epoch, identified = state.seed, state.epoch, state.identified
    else:
        raise ValueError("optimizer does not support checkpoint restoration")

    if bool(identified) != bool(state.identified):
        raise ValueError("frozen state regional mode does not match template state")

    t = next_opt.telemetry()
    telem = Telemetry(
        dt_fit=t.dt_fit,
        dt_gen=t.dt_gen,
        dt_sel=t.dt_sel,
        dt_tell=t.dt_tell,
        num_candidates=int(t.num_candidates),
    )
    raw_pending = next_opt.pending_ids()
    pending = None if not raw_pending else tuple(int(i) for i in raw_pending)
    return OptimizerState(
        dim=state.dim,
        bounds=state.bounds,
        config=state.config,
        seed=int(seed),
        epoch=int(epoch),
        spec=state.spec,
        identified=bool(identified),
        num_observations=int(next_opt.num_observations()),
        num_regions=int(next_opt.num_regions()),
        tr_length=float(next_opt.tr_length()),
        init_progress=next_opt.init_progress(),
        pending_ids=pending,
        ingestion_failure=getattr(next_opt, "ingestion_failure", None),
        telemetry=telem,
        _opt=next_opt,
        _log=(),
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
