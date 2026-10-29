from __future__ import annotations

import pytest

from baux.turbo import config as cfg
from baux.turbo.config import (
    AcquisitionKind,
    CandidateConfig,
    CandidateSampler,
    DrawAcquisitionConfig,
    DistanceMetric,
    SurrogateFit,
    NeighborIndex,
    ENNSurrogate,
    GaussianProcess,
    InitConfig,
    LHDOnlyInit,
    MorboTrustRegion,
    MultiObjectiveConfig,
    MultiTrustRegionPolicy,
    NDSOptimizerConfig,
    NoSurrogate,
    NoTrustRegion,
    ObservationHistoryConfig,
    OptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    TurboTrustRegion,
    UCBAcquisitionConfig,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)


def test_021():
    with pytest.raises(ValueError, match="init_strategy must be"):
        InitConfig(init_strategy="invalid")


def test_022():
    with pytest.raises(ValueError, match="num_init must be > 0"):
        InitConfig(num_init=0)


def test_023():
    cfg = ENNSurrogate()
    assert cfg.k is None
    assert cfg.num_samples is None
    assert cfg.num_candidates is None
    assert cfg.scale_x is False


def test_024():
    cfg = ENNSurrogate(k=10, fit=SurrogateFit(num_samples=50), scale_x=True)
    assert cfg.k == 10
    assert cfg.num_samples == 50
    assert cfg.scale_x is True


def test_025():
    with pytest.raises(
        ValueError, match=r"scale_x=True is not compatible with BPANN_DISK"
    ):
        ENNSurrogate(scale_x=True, index_driver=NeighborIndex.BPANN_DISK)


def test_026():
    cfg = SurrogateFit()
    assert cfg.num_samples is None
    assert cfg.num_candidates is None


def test_ennfitconfigcustom():
    cfg = SurrogateFit(num_samples=50, num_candidates=100)
    assert cfg.num_samples == 50
    assert cfg.num_candidates == 100


def test_027():
    with pytest.raises(ValueError, match="num_samples must be > 0"):
        SurrogateFit(num_samples=0)


def test_028():
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        SurrogateFit(num_candidates=0)


def test_029():
    cfg = ENNSurrogate(fit=SurrogateFit(num_candidates=100))
    assert cfg.num_candidates == 100


def test_030():
    acq = UCBAcquisitionConfig()
    assert acq.beta == 2.0


def test_031():
    cfg = OptimizerConfig()
    assert isinstance(cfg.trust_region, TurboTrustRegion)
    assert cfg.candidates.sampler == CandidateSampler.SOBOL
    assert cfg.init.num_init is None
    assert isinstance(cfg.surrogate, NoSurrogate)
    assert isinstance(cfg.observation_history, ObservationHistoryConfig)


def test_032():
    with pytest.raises(
        ValueError, match="init_strategy='lhd_only' requires NoSurrogate"
    ):
        OptimizerConfig(
            init=InitConfig(init_strategy=LHDOnlyInit()),
            surrogate=GaussianProcess(),
        )
    config = OptimizerConfig(
        init=InitConfig(init_strategy=LHDOnlyInit()),
        trust_region=MorboTrustRegion(
            multi_objective=MultiObjectiveConfig(num_metrics=2)
        ),
        surrogate=NoSurrogate(),
    )
    assert config.num_metrics == 2


def test_033():
    with pytest.raises(
        ValueError,
        match="NoSurrogate is not compatible with DrawAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogate(),
            acquisition=DrawAcquisitionConfig(),
        )


def test_034():
    with pytest.raises(
        ValueError,
        match="NoSurrogate is not compatible with UCBAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogate(),
            acquisition=UCBAcquisitionConfig(),
        )


def test_035():
    with pytest.raises(
        ValueError, match="ParetoAcquisitionConfig requires NDSOptimizerConfig"
    ):
        OptimizerConfig(
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )


def test_036():
    cfg = turbo_one()
    assert isinstance(cfg.trust_region, TurboTrustRegion)
    assert isinstance(cfg.surrogate, GaussianProcess)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_037():
    cfg = turbo_one(acquisition=AcquisitionKind.PARETO)
    assert isinstance(cfg.surrogate, GaussianProcess)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_038():
    cfg = turbo_one(acquisition=AcquisitionKind.UCB)
    assert isinstance(cfg.surrogate, GaussianProcess)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_039():
    cfg = turbo_one(acquisition=AcquisitionKind.THOMPSON)
    assert isinstance(cfg.surrogate, GaussianProcess)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_040():
    cfg = turbo_zero()
    assert isinstance(cfg.trust_region, TurboTrustRegion)
    assert isinstance(cfg.surrogate, NoSurrogate)
    assert isinstance(cfg.acquisition, RandomAcquisitionConfig)


def test_041():
    cfg = turbo_enn(acquisition=AcquisitionKind.PARETO)
    assert isinstance(cfg.surrogate, ENNSurrogate)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_042():
    cfg = turbo_enn(
        acquisition=AcquisitionKind.UCB,
        enn=ENNSurrogate(fit=SurrogateFit(num_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogate)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_043():
    cfg = turbo_enn(
        acquisition=AcquisitionKind.THOMPSON,
        enn=ENNSurrogate(fit=SurrogateFit(num_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogate)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_044():
    with pytest.raises(ValueError, match="num_samples required"):
        turbo_enn(acquisition=AcquisitionKind.UCB)


def test_045():
    cfg = lhd_only()
    assert isinstance(cfg.trust_region, NoTrustRegion)
    assert isinstance(cfg.init.init_strategy, LHDOnlyInit)
    assert isinstance(cfg.surrogate, NoSurrogate)


def test_046():
    cfg = turbo_enn(
        enn=ENNSurrogate(k=15),
        candidates=CandidateConfig(num_candidates=200),
        num_init=10,
    )
    assert cfg.surrogate.k == 15
    assert cfg.candidates.resolve_candidates(num_dim=3, num_arms=7) == 200
    assert cfg.init.num_init == 10


def test_047():
    cfg_turbo = OptimizerConfig(trust_region=TurboTrustRegion())
    assert cfg_turbo.num_metrics is None
    cfg_morbo = OptimizerConfig(
        trust_region=MorboTrustRegion(
            multi_objective=MultiObjectiveConfig(num_metrics=2)
        )
    )
    assert cfg_morbo.num_metrics == 2
    cfg_none = OptimizerConfig(trust_region=NoTrustRegion())
    assert cfg_none.num_metrics is None


def test_048():
    config = cfg.turbo_enn(enn=cfg.ENNSurrogate(fit=cfg.SurrogateFit(num_samples=4)))

    assert isinstance(config, cfg.OptimizerConfig)
    assert config.candidates == cfg.CandidateConfig()
    assert config.raasp_driver is cfg.RAASPDriver.ORIG


def test_enumvalues():
    assert DistanceMetric.SQUARED_L2.value == "squared_l2"
    assert DistanceMetric.COSINE.value == "cosine"


def test_multitrlengths():
    tr = MultiTrustRegionPolicy()
    assert (tr.length_init, tr.length_min, tr.length_max) == (
        tr.length.length_init,
        tr.length.length_min,
        tr.length.length_max,
    )
