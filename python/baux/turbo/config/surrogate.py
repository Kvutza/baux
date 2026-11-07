from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from baux.index import (
    INDEX_NAMES,
    DistanceMetric,
    NeighborIndex,
    NoSurrogate,
    Storage,
)


@dataclass(frozen=True)
class GaussianProcess:
    pass


@dataclass(frozen=True)
class SurrogateFit:
    num_samples: int | None = None
    num_candidates: int | None = None
    infer_aleatoric_variance_scale: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("num_samples", self.num_samples),
            ("num_candidates", self.num_candidates),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class ENNSurrogate:
    k: int | None = None
    fit: SurrogateFit = field(default_factory=SurrogateFit)
    scale_x: bool = False
    index_driver: NeighborIndex = NeighborIndex.FLAT
    distance_metric: DistanceMetric = DistanceMetric.SQUARED_L2
    enn_storage: str | None = None
    work_dir: str | os.PathLike[str] | None = None
    y_bounds: np.ndarray | None = None
    metric_learning: Any = "none"
    tied_dims: tuple[tuple[int, ...], ...] | None = None
    unscaled_dims: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        ml = str(getattr(self.metric_learning, "value", self.metric_learning)).lower()
        if ml not in ("none", "auto"):
            raise ValueError(f"Unknown metric_learning: {self.metric_learning!r}")
        if ml == "auto" and self.scale_x:
            raise ValueError("metric_learning='auto' requires scale_x=False")
        if self.y_bounds is not None:
            bounds = np.asarray(self.y_bounds, dtype=float)
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(
                    f"y_bounds must have shape (metrics, 2), got {bounds.shape}"
                )
            object.__setattr__(self, "y_bounds", bounds)

    @property
    def num_samples(self) -> int | None:
        return self.fit.num_samples

    @property
    def num_candidates(self) -> int | None:
        return self.fit.num_candidates


@dataclass(frozen=True)
class ExternalSurrogate:
    """Create independent providers implementing ``fit``, ``predict``, and ``draw``."""

    factory: Callable[[], Any]
    fit_steps: int = 50

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("surrogate factory must be callable")
        if self.fit_steps <= 0:
            raise ValueError("fit_steps must be positive")


Surrogate = NoSurrogate | GaussianProcess | ENNSurrogate | ExternalSurrogate

__all__ = [
    "DistanceMetric",
    "ENNSurrogate",
    "ExternalSurrogate",
    "GaussianProcess",
    "INDEX_NAMES",
    "NeighborIndex",
    "NoSurrogate",
    "Storage",
    "Surrogate",
    "SurrogateFit",
]
