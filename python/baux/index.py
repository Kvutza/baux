"""Nearest-neighbor index drivers, distance metrics, and storage backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DistanceMetric(Enum):
    """Distance metric used by nearest-neighbor surrogate indices."""

    SQUARED_L2 = "squared_l2"
    COSINE = "cosine"


class NeighborIndex(Enum):
    """Nearest-neighbor search backend driver selection."""

    FLAT = auto()
    AUTO = auto()
    USEARCH = auto()
    BPANN_DISK = auto()
    METAL = auto()
    OPENCL = auto()
    CUDA = auto()


class Storage(Enum):
    """Observation storage medium for nearest-neighbor indices."""

    MEMORY = "memory"
    DISK = "disk"


@dataclass(frozen=True)
class NoSurrogate:
    """Marker configuration disabling surrogate modeling."""

    pass


INDEX_NAMES: dict[NeighborIndex, str] = {
    NeighborIndex.FLAT: "exact",
    NeighborIndex.AUTO: "auto",
    NeighborIndex.USEARCH: "usearch",
    NeighborIndex.BPANN_DISK: "bpann_disk",
    NeighborIndex.METAL: "metal",
    NeighborIndex.OPENCL: "opencl",
    NeighborIndex.CUDA: "cuda",
}

__all__ = [
    "DistanceMetric",
    "INDEX_NAMES",
    "NeighborIndex",
    "NoSurrogate",
    "Storage",
]
