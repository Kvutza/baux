from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .acquisition import (
    AcquisitionConfig,
    DrawAcquisitionConfig,
    ParetoAcquisitionConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .candidates import RestartCenterPolicy, RestartCenterSelection
from .morbo_region import (
    MultiObjectiveConfig,
    Rescalarize,
    RescalePolicy,
    TrustRegionBounds,
)

AllocationPolicy = Literal["reward_ucb", "balanced"]
RestartMemoryPolicy = Literal["fresh", "retain_relevant"]
RegionMethod = Literal["turbo", "fixed_length"]


def validate_acquisitions(
    num_regions: int, acquisitions: tuple[AcquisitionConfig | None, ...]
) -> None:
    if not isinstance(acquisitions, tuple):
        raise TypeError("region_acquisitions must be a tuple")
    if acquisitions and len(acquisitions) != num_regions:
        raise ValueError("region_acquisitions must be empty or match num_regions")
    for acquisition in acquisitions:
        if acquisition is None:
            continue
        if isinstance(acquisition, ParetoAcquisitionConfig):
            raise ValueError("Pareto regional acquisition is unsupported")
        if not isinstance(
            acquisition,
            (UCBAcquisitionConfig, DrawAcquisitionConfig, RandomAcquisitionConfig),
        ):
            raise TypeError(
                "each regional acquisition must be an AcquisitionConfig or None"
            )
        if isinstance(acquisition, UCBAcquisitionConfig) and (
            not isfinite(acquisition.beta) or acquisition.beta <= 0
        ):
            raise ValueError("regional UCB beta must be finite and > 0")


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
    allocation_policy: AllocationPolicy = "reward_ucb"
    restart_memory_policy: RestartMemoryPolicy = "fresh"
    restart_center: RestartCenterPolicy = RestartCenterPolicy()
    region_method: RegionMethod = "turbo"
    region_acquisitions: tuple[AcquisitionConfig | None, ...] = ()

    def __post_init__(self) -> None:
        if self.allocation_policy not in ("reward_ucb", "balanced"):
            raise ValueError("unknown regional allocation policy")
        if self.restart_memory_policy not in ("fresh", "retain_relevant"):
            raise ValueError("unknown regional restart memory policy")
        if not isinstance(self.restart_center, RestartCenterPolicy):
            raise TypeError("restart_center must be a RestartCenterPolicy")
        if self.region_method not in ("turbo", "fixed_length"):
            raise ValueError("unknown regional method")
        validate_acquisitions(self.num_regions, self.region_acquisitions)

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
    "AllocationPolicy",
    "MorboTrustRegion",
    "MultiObjectiveConfig",
    "MultiTrustRegionPolicy",
    "NoTrustRegion",
    "Rescalarize",
    "RescalePolicy",
    "RegionMethod",
    "RestartCenterPolicy",
    "RestartCenterSelection",
    "RestartMemoryPolicy",
    "TrustRegionBounds",
    "TrustRegionConfig",
    "TurboTrustRegion",
]
