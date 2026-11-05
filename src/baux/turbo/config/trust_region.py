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

AllocationPolicy = Literal["incumbent_ucb1", "least_trials", "joint_acquisition"]
RestartMemoryPolicy = Literal["fresh", "retain_relevant"]
RegionController = Literal["turbo", "fixed_radius"]


def validate_acquisitions(
    count: int, acquisitions: tuple[AcquisitionConfig | None, ...]
) -> None:
    if not isinstance(acquisitions, tuple):
        raise TypeError("acquisitions must be a tuple")
    if acquisitions and len(acquisitions) != count:
        raise ValueError("acquisitions must be empty or match region count")
    for acquisition in acquisitions:
        if acquisition is None:
            continue
        if not isinstance(
            acquisition,
            (
                UCBAcquisitionConfig,
                DrawAcquisitionConfig,
                RandomAcquisitionConfig,
                ParetoAcquisitionConfig,
            ),
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
class RegionalSearchPolicy:
    count: int = 4
    length: TrustRegionBounds = TrustRegionBounds()
    success_tolerance: int = 3
    failure_tolerance: int = 5
    sharing: Literal[
        "box_contain", "ellipsoid_ball", "nearest_center", "origin_only"
    ] = "box_contain"
    allocation: AllocationPolicy = "incumbent_ucb1"
    restart_memory: RestartMemoryPolicy = "fresh"
    restart_center: RestartCenterPolicy = RestartCenterPolicy()
    controller: RegionController = "turbo"
    acquisitions: tuple[AcquisitionConfig | None, ...] = ()

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("regional search count must be positive")
        if self.success_tolerance <= 0 or self.failure_tolerance <= 0:
            raise ValueError("regional search tolerances must be positive")
        if self.allocation not in (
            "incumbent_ucb1",
            "least_trials",
            "joint_acquisition",
        ):
            raise ValueError("unknown regional allocation policy")
        if self.restart_memory not in ("fresh", "retain_relevant"):
            raise ValueError("unknown regional restart memory policy")
        if not isinstance(self.restart_center, RestartCenterPolicy):
            raise TypeError("restart_center must be a RestartCenterPolicy")
        if self.controller not in ("turbo", "fixed_radius"):
            raise ValueError("unknown regional controller")
        validate_acquisitions(self.count, self.acquisitions)

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
    NoTrustRegion | TurboTrustRegion | MorboTrustRegion | RegionalSearchPolicy
)

__all__ = [
    "AllocationPolicy",
    "MorboTrustRegion",
    "MultiObjectiveConfig",
    "RegionalSearchPolicy",
    "NoTrustRegion",
    "Rescalarize",
    "RescalePolicy",
    "RegionController",
    "RestartCenterPolicy",
    "RestartCenterSelection",
    "RestartMemoryPolicy",
    "TrustRegionBounds",
    "TrustRegionConfig",
    "TurboTrustRegion",
]
