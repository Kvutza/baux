"""Data containers and planning helpers for baux optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .turbo.config import OptimizerConfig
from .turbo.config.encode import enn_k, is_lhd
from .turbo.config.surrogate import (
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    NoSurrogate,
)
from .turbo.optimizer_types import Telemetry, _Obs, _Plan


@dataclass(frozen=True, order=True)
class AskId:
    """Opaque, process-wide identity for one regional evaluation.

    The optimizer checks this token against the proposal's stored generation
    provenance when completing or cancelling it. Pass it back unchanged.
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("AskId.value must be an integer")
        if self.value < 0 or self.value >= 2**64:
            raise ValueError("AskId.value must fit in an unsigned 64-bit integer")


@dataclass(frozen=True)
class AskBatch:
    """Coordinates, evaluation identities, and structured PyTree views from one ask."""

    ids: tuple[AskId, ...]
    points: np.ndarray
    spec: Any = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[0] != len(self.ids):
            raise ValueError("AskBatch points must have one row per evaluation ID")

    @property
    def params(self) -> tuple[Any, ...]:
        """Return structured parameter PyTrees (one per arm) when initialized with a PyTreeSpec."""
        if self.spec is not None:
            return self.spec.unflatten_batch(self.points, weights=self.weights)
        return tuple(self.points[i] for i in range(self.points.shape[0]))

    @property
    def stacked(self) -> Any:
        """Return a single PyTree whose leaves have leading batch dimension `(num_arms, *shape)`."""
        if self.spec is not None:
            return self.spec.unflatten_stacked(self.points, weights=self.weights)
        return self.points

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.points, dtype=dtype)


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


def _plan(cfg: OptimizerConfig) -> _Plan:
    if is_lhd(cfg):
        return _Plan("lhd")
    if isinstance(cfg.surrogate, ENNSurrogate):
        return _Plan("enn", enn_k(cfg))
    if isinstance(cfg.surrogate, NoSurrogate):
        return _Plan("zero")
    if isinstance(cfg.surrogate, GaussianProcess):
        import sys

        if sys.platform == "emscripten":
            raise RuntimeError(
                "GaussianProcess / baux.gp() requires PyTorch + GPyTorch and is not enabled "
                "in browser WASM; use baux.enn() or baux.zero(), "
                "or switch Runtime to Local Native."
            )
        from .turbo.gp import Surrogate

        return _Plan("gp", gp=Surrogate())
    if isinstance(cfg.surrogate, ExternalSurrogate):
        return _Plan(
            "gp", gp=cfg.surrogate.factory(), fit_steps=cfg.surrogate.fit_steps
        )
    raise ValueError(f"unsupported surrogate: {type(cfg.surrogate)!r}")


__all__ = [
    "AskBatch",
    "AskId",
    "Telemetry",
    "_obs",
    "_plan",
]
