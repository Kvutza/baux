from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .neighbor_index import (
    INDEX_NAMES,
    DistanceMetric,
    GaussianProcess,
    NeighborIndex,
    NoSurrogate,
)


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

    def __post_init__(self) -> None:
        if self.scale_x and self.index_driver == NeighborIndex.BPANN_DISK:
            raise ValueError("scale_x=True is not compatible with BPANN_DISK")
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


Surrogate = NoSurrogate | GaussianProcess | ENNSurrogate

__all__ = [
    "DistanceMetric",
    "ENNSurrogate",
    "GaussianProcess",
    "INDEX_NAMES",
    "NeighborIndex",
    "NoSurrogate",
    "Surrogate",
    "SurrogateFit",
]
