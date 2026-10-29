from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .morbo_region import (
    MultiObjectiveConfig,
    Rescalarize,
    RescalePolicy,
    TrustRegionBounds,
)


@dataclass(frozen=True)
class NoTrustRegion:
    noise_aware: bool = False


@dataclass(frozen=True)
class TurboTrustRegion:
    length: TrustRegionBounds = TrustRegionBounds()
    noise_aware: bool = False

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MorboTrustRegion:
    multi_objective: MultiObjectiveConfig
    length: TrustRegionBounds = TrustRegionBounds()
    rescale_policy: RescalePolicy = RescalePolicy()
    noise_aware: bool = False

    @property
    def rescalarize(self) -> Rescalarize:
        return self.rescale_policy.rescalarize

    @property
    def num_metrics(self) -> int:
        return self.multi_objective.num_metrics

    @property
    def alpha(self) -> float:
        return self.multi_objective.alpha

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


@dataclass(frozen=True)
class MultiTrustRegionPolicy:
    num_regions: int = 4
    length: TrustRegionBounds = TrustRegionBounds()
    succ_tolerance: int = 3
    fail_tolerance: int = 5
    sharing_policy: Literal["shared", "nearest_center", "independent"] = "shared"

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max


TrustRegionConfig = (
    NoTrustRegion | TurboTrustRegion | MorboTrustRegion | MultiTrustRegionPolicy
)

__all__ = [
    "MorboTrustRegion",
    "MultiObjectiveConfig",
    "MultiTrustRegionPolicy",
    "NoTrustRegion",
    "Rescalarize",
    "RescalePolicy",
    "TrustRegionBounds",
    "TrustRegionConfig",
    "TurboTrustRegion",
]
