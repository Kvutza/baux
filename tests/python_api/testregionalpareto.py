from __future__ import annotations

import numpy as np
from baux import ask, complete, init, tell
from baux.turbo.config import (
    ENNSurrogate,
    InitConfig,
    MorboTrustRegion,
    MultiObjectiveConfig,
    NDSOptimizerConfig,
    OptimizerConfig,
    ParetoAcquisitionConfig,
    RegionalSearchPolicy,
    TrustRegionBounds,
)
from baux.turbo.config.encode import encode


def testregionalpareto():
    acquisitions = (ParetoAcquisitionConfig(), ParetoAcquisitionConfig())
    policy = RegionalSearchPolicy(count=2, acquisitions=acquisitions)
    assert policy.acquisitions == acquisitions

    config = OptimizerConfig(
        init=InitConfig(num_init=2),
        surrogate=ENNSurrogate(),
        acquisition=ParetoAcquisitionConfig(),
        acq_optimizer=NDSOptimizerConfig(),
        trust_region=policy,
    )
    cfg_encoded = encode(config)
    assert cfg_encoded["region_acquisitions"] == ["pareto", "pareto"]

    # Test via pure functional baux.init / baux.ask / baux.tell / baux.complete
    state = init(
        domain=np.array([[0.0, 1.0]]),
        config=config,
        seed=85,
    )
    assert state.num_regions == 2
    state, batch = ask(state, 2)
    assert len(batch.ids) == 2
    points = batch.points
    values = np.column_stack((points[:, 0], 1.0 - points[:, 0]))

    # Pure functional tell with multi-objective values
    state, _ = tell(state, points, values)
    assert state.num_observations == 2

    # Next ask produces batch from regional Pareto policy
    state, next_batch = ask(state, 2)
    assert next_batch.points.shape == (2, 1)
    assert len(next_batch.ids) == 2


def test_regionalcomplete():
    policy = RegionalSearchPolicy(count=2)
    config = OptimizerConfig(
        init=InitConfig(num_init=1),
        surrogate=ENNSurrogate(),
        trust_region=policy,
    )
    state = init(
        domain=np.array([[0.0, 1.0]]),
        config=config,
        seed=42,
    )
    state, batch = ask(state, 2)
    assert len(batch.ids) == 2
    target_id = batch.ids[0]
    state = complete(state, [target_id], np.array([[0.5]]))
    assert target_id.value not in (state.pending_ids or ())


def test_morbotrust():
    morbo = MorboTrustRegion(
        multi_objective=MultiObjectiveConfig(num_metrics=2, alpha=0.05),
        length=TrustRegionBounds(length_init=0.5, length_min=0.01, length_max=1.0),
    )
    assert morbo.num_metrics == 2
    assert morbo.alpha == 0.05

    config = OptimizerConfig(
        init=InitConfig(num_init=2),
        surrogate=ENNSurrogate(),
        acquisition=ParetoAcquisitionConfig(),
        acq_optimizer=NDSOptimizerConfig(),
        trust_region=morbo,
    )
    cfg_encoded = encode(config)
    assert cfg_encoded["trust_region"] == "morbo"
    assert cfg_encoded["num_metrics"] == 2
    assert cfg_encoded["alpha"] == 0.05

    state = init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=42,
    )
    state, batch = ask(state, 2)
    pts = batch.points
    vals = np.column_stack((pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]))
    state, _ = tell(state, pts, vals)
    assert state.num_observations == 2
