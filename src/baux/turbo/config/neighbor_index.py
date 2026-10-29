from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DistanceMetric(Enum):
    SQUARED_L2 = "squared_l2"
    COSINE = "cosine"


class NeighborIndex(Enum):
    FLAT = auto()
    AUTO = auto()
    USEARCH = auto()
    BPANN_DISK = auto()
    METAL = auto()
    OPENCL = auto()
    CUDA = auto()


INDEX_NAMES: dict[NeighborIndex, str] = {
    NeighborIndex.FLAT: "exact",
    NeighborIndex.AUTO: "auto",
    NeighborIndex.USEARCH: "usearch",
    NeighborIndex.BPANN_DISK: "bpann_disk",
    NeighborIndex.METAL: "metal",
    NeighborIndex.OPENCL: "opencl",
    NeighborIndex.CUDA: "cuda",
}


@dataclass(frozen=True)
class GaussianProcess:
    pass


@dataclass(frozen=True)
class NoSurrogate:
    pass
