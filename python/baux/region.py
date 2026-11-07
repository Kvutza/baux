from __future__ import annotations

from baux._rust import Portfolio, Region, RegionTrial, Trial
from baux.experimental.multi_region import SharingPolicy
from baux.turbo.config.trust_region import (
    AllocationPolicy,
    MorboTrustRegion,
    MultiObjectiveConfig,
    NoTrustRegion,
    RegionalSearchPolicy,
    RegionController,
    Rescalarize,
    RescalePolicy,
    RestartCenterPolicy,
    RestartCenterSelection,
    RestartMemoryPolicy,
    TrustRegionBounds,
    TrustRegionConfig,
    TurboTrustRegion,
)

TurboRegion = TurboTrustRegion
MorboRegion = MorboTrustRegion
GlobalRegion = NoTrustRegion
RegionBounds = TrustRegionBounds
RegionPolicy = RegionalSearchPolicy
RegionConfig = TrustRegionConfig

__all__ = [
    "AllocationPolicy",
    "GlobalRegion",
    "MorboRegion",
    "MorboTrustRegion",
    "MultiObjectiveConfig",
    "NoTrustRegion",
    "Portfolio",
    "Region",
    "RegionBounds",
    "RegionConfig",
    "RegionController",
    "RegionPolicy",
    "RegionTrial",
    "RegionalSearchPolicy",
    "Rescalarize",
    "RescalePolicy",
    "RestartCenterPolicy",
    "RestartCenterSelection",
    "RestartMemoryPolicy",
    "SharingPolicy",
    "Trial",
    "TrustRegionBounds",
    "TrustRegionConfig",
    "TurboRegion",
    "TurboTrustRegion",
]
