from __future__ import annotations

from .acquisition import (
    AcqOptimizerConfig,
    AcquisitionConfig,
    AcquisitionKind,
    DrawAcquisitionConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .candidates import CandidateConfig, CandidateSampler
from .initialization import InitConfig, LHDOnlyInit
from .optimizer import OptimizerConfig
from .surrogate import ENNSurrogate, GaussianProcess, NoSurrogate
from .trust_region import NoTrustRegion, TrustRegionConfig, TurboTrustRegion


def _candidates(
    sampler: CandidateSampler,
    num_candidates: int | None,
    *,
    num_candidates_per_arm: int | None = None,
) -> CandidateConfig:
    return CandidateConfig(
        sampler=sampler,
        num_candidates=num_candidates,
        num_candidates_per_arm=num_candidates_per_arm,
    )


def _acqconfigs(
    acquisition: AcquisitionKind,
) -> tuple[AcquisitionConfig, AcqOptimizerConfig]:
    if acquisition == AcquisitionKind.PARETO:
        return ParetoAcquisitionConfig(), NDSOptimizerConfig()
    if acquisition == AcquisitionKind.UCB:
        return UCBAcquisitionConfig(), RAASPOptimizerConfig()
    if acquisition == AcquisitionKind.THOMPSON:
        return DrawAcquisitionConfig(), RAASPOptimizerConfig()
    raise ValueError(
        f"acquisition must be AcquisitionKind.THOMPSON, AcquisitionKind.PARETO, or AcquisitionKind.UCB, got {acquisition!r}"
    )


def turbo_one(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    sampler: CandidateSampler = CandidateSampler.SOBOL,
    acquisition: AcquisitionKind = AcquisitionKind.THOMPSON,
) -> OptimizerConfig:
    acq_config, acq_optimizer = _acqconfigs(acquisition)
    return OptimizerConfig(
        trust_region=trust_region or TurboTrustRegion(),
        candidates=_candidates(sampler, num_candidates),
        init=InitConfig(num_init=num_init),
        surrogate=GaussianProcess(),
        acquisition=acq_config,
        acq_optimizer=acq_optimizer,
    )


def turbo_zero(
    *,
    num_candidates: int | None = None,
    num_candidates_per_arm: int | None = None,
    num_init: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    sampler: CandidateSampler = CandidateSampler.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or TurboTrustRegion(),
        candidates=_candidates(
            sampler,
            num_candidates,
            num_candidates_per_arm=num_candidates_per_arm,
        ),
        init=InitConfig(num_init=num_init),
        surrogate=NoSurrogate(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
    )


def turbo_enn(
    *,
    enn: ENNSurrogate | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidates: CandidateConfig | None = None,
    num_init: int | None = None,
    acquisition: AcquisitionKind = AcquisitionKind.PARETO,
) -> OptimizerConfig:
    acq_config, acq_optimizer = _acqconfigs(acquisition)
    surrogate = enn if enn is not None else ENNSurrogate()
    if surrogate.fit.num_samples is None and acquisition != AcquisitionKind.PARETO:
        raise ValueError(
            f"enn.fit.num_samples required for acquisition={acquisition!r}"
        )
    return OptimizerConfig(
        trust_region=trust_region or TurboTrustRegion(),
        candidates=candidates or CandidateConfig(),
        init=InitConfig(num_init=num_init),
        surrogate=surrogate,
        acquisition=acq_config,
        acq_optimizer=acq_optimizer,
    )


def lhd_only(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    sampler: CandidateSampler = CandidateSampler.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or NoTrustRegion(),
        candidates=_candidates(sampler, num_candidates),
        init=InitConfig(init_strategy=LHDOnlyInit(), num_init=num_init),
        surrogate=NoSurrogate(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
    )
