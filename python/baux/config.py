"""Public configuration re-exports for BAUX optimizers, surrogates, and trust regions."""

from baux import region as _region
from baux.index import (
    INDEX_NAMES,
    DistanceMetric,
    NeighborIndex,
    Storage,
)
from baux.surrogate import (
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    NoSurrogate,
    Surrogate,
    SurrogateFit,
)
from baux.turbo import config as _turbo_config
from baux.turbo.config import (
    AcquisitionConfig,
    AcquisitionKind,
    CandidateConfig,
    CandidateSampler,
    InitConfig,
    InitStrategy,
    OptimizerConfig,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)

AllocationPolicy = _region.AllocationPolicy
AcqOptimizerConfig = _turbo_config.AcqOptimizerConfig
DrawAcquisitionConfig = _turbo_config.DrawAcquisitionConfig
HybridInit = _turbo_config.HybridInit
LHDOnlyInit = _turbo_config.LHDOnlyInit
MorboTrustRegion = _region.MorboTrustRegion
MultiObjectiveConfig = _region.MultiObjectiveConfig
NDSOptimizerConfig = _turbo_config.NDSOptimizerConfig
NoTrustRegion = _region.NoTrustRegion
ObservationHistoryConfig = _turbo_config.ObservationHistoryConfig
ParetoAcquisitionConfig = _turbo_config.ParetoAcquisitionConfig
Portfolio = _region.Portfolio
RAASPDriver = _turbo_config.RAASPDriver
RAASPOptimizerConfig = _turbo_config.RAASPOptimizerConfig
RandomAcquisitionConfig = _turbo_config.RandomAcquisitionConfig
Region = _region.Region
RegionController = _region.RegionController
RegionTrial = _region.RegionTrial
RegionalSearchPolicy = _region.RegionalSearchPolicy
Rescalarize = _region.Rescalarize
RescalePolicy = _region.RescalePolicy
RestartCenterPolicy = _region.RestartCenterPolicy
RestartCenterSelection = _region.RestartCenterSelection
RestartMemoryPolicy = _region.RestartMemoryPolicy
SharingPolicy = _region.SharingPolicy
Trial = _region.Trial
TrustRegionBounds = _region.TrustRegionBounds
TrustRegionConfig = _region.TrustRegionConfig
TurboTrustRegion = _region.TurboTrustRegion
UCBAcquisitionConfig = _turbo_config.UCBAcquisitionConfig
default_candidates = _turbo_config.default_candidates
fixed_candidates = _turbo_config.fixed_candidates

__all__ = [
    "AcqOptimizerConfig",
    "AcquisitionConfig",
    "AcquisitionKind",
    "AllocationPolicy",
    "CandidateConfig",
    "CandidateSampler",
    "DistanceMetric",
    "DrawAcquisitionConfig",
    "ENNSurrogate",
    "ExternalSurrogate",
    "GaussianProcess",
    "HybridInit",
    "INDEX_NAMES",
    "InitConfig",
    "InitStrategy",
    "LHDOnlyInit",
    "MorboTrustRegion",
    "MultiObjectiveConfig",
    "NDSOptimizerConfig",
    "NeighborIndex",
    "NoSurrogate",
    "NoTrustRegion",
    "ObservationHistoryConfig",
    "OptimizerConfig",
    "ParetoAcquisitionConfig",
    "Portfolio",
    "RAASPDriver",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "Region",
    "RegionController",
    "RegionTrial",
    "RegionalSearchPolicy",
    "Rescalarize",
    "RescalePolicy",
    "RestartCenterPolicy",
    "RestartCenterSelection",
    "RestartMemoryPolicy",
    "SharingPolicy",
    "Storage",
    "Surrogate",
    "SurrogateFit",
    "Trial",
    "TrustRegionBounds",
    "TrustRegionConfig",
    "TurboTrustRegion",
    "UCBAcquisitionConfig",
    "default_candidates",
    "fixed_candidates",
    "lhd_only",
    "turbo_enn",
    "turbo_one",
    "turbo_zero",
]
