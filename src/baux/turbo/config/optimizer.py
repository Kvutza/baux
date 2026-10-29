from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .acquisition import (
    AcqOptimizerConfig,
    AcquisitionConfig,
    DrawAcquisitionConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .candidates import CandidateConfig, RAASPDriver
from .initialization import InitConfig, LHDOnlyInit
from .surrogate import ENNSurrogate, GaussianProcess, NoSurrogate, Surrogate
from .trust_region import (
    MorboTrustRegion,
    TrustRegionConfig,
    TurboTrustRegion,
)

if TYPE_CHECKING:
    from .candidates import CandidateSampler


@dataclass(frozen=True)
class ObservationHistoryConfig:
    pass


def _randomacquisition() -> RandomAcquisitionConfig:
    return RandomAcquisitionConfig()


def _raaspoptimizer() -> RAASPOptimizerConfig:
    return RAASPOptimizerConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    trust_region: TrustRegionConfig = TurboTrustRegion()
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    init: InitConfig = field(default_factory=InitConfig)
    surrogate: Surrogate = NoSurrogate()
    acquisition: AcquisitionConfig = field(default_factory=_randomacquisition)
    acq_optimizer: AcqOptimizerConfig = field(default_factory=_raaspoptimizer)
    observation_history: ObservationHistoryConfig = ObservationHistoryConfig()

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def num_metrics(self) -> int | None:
        if isinstance(self.trust_region, MorboTrustRegion):
            return self.trust_region.num_metrics
        return None

    @property
    def candidate_sampler(self) -> CandidateSampler:
        return self.candidates.sampler

    @property
    def num_candidates(self) -> int | None:
        return self.candidates.num_candidates

    @property
    def raasp_driver(self) -> RAASPDriver:
        return self.candidates.raasp_driver


def _validate(config: OptimizerConfig) -> None:
    if isinstance(config.init.init_strategy, LHDOnlyInit) and not isinstance(
        config.surrogate, NoSurrogate
    ):
        raise ValueError("init_strategy='lhd_only' requires NoSurrogate")
    if isinstance(config.surrogate, NoSurrogate) and isinstance(
        config.acquisition, (DrawAcquisitionConfig, UCBAcquisitionConfig)
    ):
        name = type(config.acquisition).__name__
        raise ValueError(f"NoSurrogate is not compatible with {name}")
    if isinstance(config.acquisition, ParetoAcquisitionConfig) and not isinstance(
        config.acq_optimizer, NDSOptimizerConfig
    ):
        raise ValueError("ParetoAcquisitionConfig requires NDSOptimizerConfig")
    if not isinstance(config.surrogate, (NoSurrogate, GaussianProcess, ENNSurrogate)):
        raise TypeError(f"unsupported surrogate: {type(config.surrogate)!r}")
