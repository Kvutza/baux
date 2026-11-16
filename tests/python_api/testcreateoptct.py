from __future__ import annotations

import inspect
from dataclasses import replace

import baux
import numpy as np
import pytest
from baux import AskBatch, AskId, Telemetry
from baux.turbo.config import (
    NeighborIndex,
    RandomAcquisitionConfig,
    RegionalSearchPolicy,
    RestartCenterPolicy,
    TrustRegionBounds,
    UCBAcquisitionConfig,
    lhd_only,
    turbo_enn,
    turbo_one,
    turbo_zero,
)


def test_factorysignature():
    init_params = set(inspect.signature(baux.init).parameters)
    assert {"domain", "config", "seed"}.issubset(init_params)
    ask_params = set(inspect.signature(baux.ask).parameters)
    assert {"state", "num_arms"}.issubset(ask_params)
    tell_params = set(inspect.signature(baux.tell).parameters)
    assert {"state", "x", "y"}.issubset(tell_params)


@pytest.mark.parametrize(
    "cfg",
    [
        turbo_enn(num_init=2),
        turbo_zero(num_init=2),
        lhd_only(num_init=2),
    ],
)
def test_optimizercontract(cfg):
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    state = baux.init(domain=bounds, config=cfg, seed=42)
    assert isinstance(state, baux.OptimizerState)
    next_state, batch = baux.ask(state, 2)

    assert isinstance(batch, AskBatch)
    assert batch.points.shape == (2, 2)
    assert np.all((bounds[:, 0] <= batch.points) & (batch.points <= bounds[:, 1]))
    assert isinstance(state.telemetry, Telemetry)
    assert isinstance(next_state.telemetry, Telemetry)


def test_surrogate():
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    enn_state = baux.init(
        domain=bounds,
        config=turbo_enn(num_init=2),
        seed=42,
    )
    zero_state = baux.init(
        domain=bounds,
        config=turbo_zero(num_init=2),
        seed=42,
    )

    assert enn_state._opt.has_surrogate()
    assert not zero_state._opt.has_surrogate()
    s_enn, batch = baux.ask(enn_state, 2)
    s_enn, estimate = baux.tell(s_enn, batch.points, -np.sum(batch.points**2, axis=1))
    assert estimate.shape == (2,)
    assert np.all(np.isfinite(estimate))
    assert np.allclose(s_enn.x_obs, batch.points)


@pytest.mark.parametrize(
    "preset,sharing",
    [
        (preset, sharing)
        for preset in (turbo_enn, turbo_zero)
        for sharing in (
            "box_contain",
            "ellipsoid_ball",
            "nearest_center",
            "origin_only",
        )
    ],
)
def test_regionalfactory(preset, sharing):
    cfg = replace(
        preset(num_init=2),
        trust_region=RegionalSearchPolicy(
            count=3,
            length=TrustRegionBounds(length_init=0.3),
            sharing=sharing,
        ),
    )
    state = baux.init(
        domain=np.array([[-2.0, 3.0], [10.0, 20.0]]),
        config=cfg,
        seed=42,
    )
    assert state.num_regions == 3
    assert state.tr_length == pytest.approx(0.3)
    for iteration in range(4):
        if iteration > 0 and state.pending_ids:
            with pytest.raises(ValueError, match="outstanding"):
                baux.ask(state, 1)
        state, batch = baux.ask(state, 3)
        y = -np.sum(batch.points**2, axis=1)
        state, estimate = baux.tell(state, batch.points, y)
        assert estimate.shape == (3,)
        assert np.all(np.isfinite(estimate))


def test_partialcompletion():
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        seed=31,
    )
    state, warmup = baux.ask(state, 1)
    warmup_ids = state.pending_ids
    assert warmup_ids is not None and len(warmup_ids) == 1
    state = baux.complete(state, warmup.ids, np.array([0.0]))
    assert state.pending_ids is None

    state, batch = baux.ask(state, 3)
    ids = batch.ids
    assert ids is not None and len(ids) == 3
    state = baux.complete(state, ids[2:], np.array([3.0]))
    assert state.pending_ids == (ids[0].value, ids[1].value)
    with pytest.raises(ValueError, match="duplicate"):
        baux.complete(state, ids[2:], np.array([3.0]))
    state = baux.complete(state, ids[:2], np.array([1.0, 2.0]))
    assert state.pending_ids is None

    state, batch2 = baux.ask(state, 3)
    assert batch2.points.shape == (3, 2)
    next_ids = batch2.ids
    assert next_ids is not None and len(next_ids) == 3
    state = baux.cancel(state, next_ids[:2])
    state = baux.complete(state, next_ids[2:], np.array([4.0]))
    assert state.pending_ids is None


def test_typedasklifecycle():
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        seed=31,
    )
    state, batch = baux.ask(state, 1)
    assert isinstance(batch, AskBatch)
    assert len(batch.ids) == 1 and isinstance(batch.ids[0], AskId)
    assert batch.points.shape == (1, 2)
    state = baux.complete(state, batch, np.array([0.0]))
    assert state.pending_ids is None

    state, next_batch = baux.ask(state, 2)
    state = baux.cancel(state, [next_batch.ids[0]])
    state = baux.complete(state, [next_batch.ids[1]], np.array([1.0]))
    assert state.pending_ids is None


def assertobservations(
    state: baux.OptimizerState, points: np.ndarray | tuple, values: tuple
):
    observed = {
        tuple(p): float(v)
        for p, v in zip(
            state.x_obs,
            state.y_obs.reshape(-1),
            strict=True,
        )
    }
    for point, value in zip(points, values, strict=True):
        assert tuple(point) in observed
        assert observed[tuple(point)] == value


def test_overlapaskids():
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=32,
    )
    state, warmup = baux.ask(state, 1)
    state = baux.complete(state, warmup, np.array([0.0]))

    state, first = baux.ask(state, 1)
    state, second = baux.ask(state, 1)
    assert set(state.pending_ids) == {first.ids[0].value, second.ids[0].value}
    state = baux.complete(state, first, np.array([10.0]))
    assert state.pending_ids == (second.ids[0].value,)

    state, replacement = baux.ask(state, 1)
    assert replacement.ids[0] not in (*first.ids, *second.ids)
    assert set(state.pending_ids) == {second.ids[0].value, replacement.ids[0].value}
    state = baux.complete(state, second, np.array([20.0]))
    state = baux.complete(state, replacement, np.array([30.0]))
    assert state.pending_ids is None
    all_points = np.vstack([first.points, second.points, replacement.points])
    assertobservations(state, all_points, (10.0, 20.0, 30.0))


def testregionrestart():
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=3,
            allocation="least_trials",
            success_tolerance=1,
        ),
    )
    state = baux.init(
        domain=np.array([[10.0, 20.0], [-2.0, 2.0]]),
        config=config,
        seed=93,
    )
    state, warmup = baux.ask(state, 1)
    state = baux.complete(state, warmup, np.array([0.0]))
    state, group = baux.ask(state, 3)
    state = baux.complete(state, [group.ids[0]], np.array([10.0]))

    center = np.array([15.0, 0.0])
    state, abandoned = baux.restart_region(state, 1, center)
    assert center.flags.writeable
    assert abandoned == (group.ids[1],)
    assert state.pending_ids == (group.ids[2].value,)
    with pytest.raises(ValueError, match="duplicate"):
        baux.complete(state, abandoned, np.array([20.0]))

    prev_obs = state.num_observations
    state, _ = baux.import_observations(state, group.points[1:2], np.array([20.0]))
    assert state.num_observations == prev_obs + 1
    assert state.pending_ids == (group.ids[2].value,)
    state = baux.complete(state, [group.ids[2]], np.array([30.0]))
    assert state.pending_ids is None
    assertobservations(state, group.points, (10.0, 20.0, 30.0))


def testidvalidation():
    with pytest.raises(TypeError, match="integer"):
        AskId(True)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        AskId(2**64)


def testpolicyencode():
    from baux.turbo.config.encode import encode

    defaults = RegionalSearchPolicy()
    assert defaults.allocation == "incumbent_ucb1"
    assert defaults.restart_memory == "fresh"
    assert defaults.controller == "turbo"

    config = replace(
        turbo_enn(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=2,
            allocation="least_trials",
            restart_memory="retain_relevant",
            restart_center=RestartCenterPolicy(candidates=32, selection="maximin"),
            controller="fixed_radius",
            acquisitions=(None, UCBAcquisitionConfig(beta=1.25)),
        ),
    )
    encoded = encode(config)
    assert encoded["region_allocation"] == "least_trials"
    assert encoded["region_restart_memory"] == "retain_relevant"
    assert encoded["region_restart_candidates"] == 32
    assert encoded["region_restart_center_selection"] == "maximin"
    assert encoded["region_controller"] == "fixed_radius"
    assert encoded["region_acquisitions"] == [None, "ucb"]
    assert encoded["region_acquisition_betas"] == [None, 1.25]
    state = baux.init(
        domain=np.array([[0.0, 1.0]]),
        config=config,
        seed=83,
    )
    assert state.num_regions == 2

    joint_config = replace(
        turbo_enn(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=2,
            allocation="joint_acquisition",
        ),
    )
    joint_state = baux.init(
        domain=np.array([[0.0, 1.0]]),
        config=joint_config,
        seed=85,
    )
    state, batch = baux.ask(joint_state, 1)
    assert batch.points.shape == (1, 1)

    random_config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=2,
            acquisitions=(None, RandomAcquisitionConfig()),
        ),
    )
    random_state = baux.init(
        domain=np.array([[0.0, 1.0]]),
        config=random_config,
        seed=84,
    )
    state, batch2 = baux.ask(random_state, 2)
    assert batch2.points.shape == (2, 1)


def testbadpolicies():
    with pytest.raises(ValueError, match="allocation policy"):
        RegionalSearchPolicy(allocation="unknown")
    with pytest.raises(ValueError, match="restart memory policy"):
        RegionalSearchPolicy(restart_memory="unknown")
    with pytest.raises(ValueError, match="regional controller"):
        RegionalSearchPolicy(controller="dynamic")
    with pytest.raises(ValueError, match="match region count"):
        RegionalSearchPolicy(count=2, acquisitions=(None,))
    with pytest.raises(ValueError, match="beta"):
        RegionalSearchPolicy(
            acquisitions=(UCBAcquisitionConfig(beta=float("inf")),) * 4
        )


def test_badregionconfig():
    with pytest.raises(ValueError, match="count"):
        RegionalSearchPolicy(count=0)


@pytest.mark.parametrize(
    "sharing",
    ["box_contain", "ellipsoid_ball", "nearest_center", "origin_only"],
)
def test_diskregions(tmp_path, sharing):
    base = turbo_enn(num_init=2)
    cfg = replace(
        base,
        surrogate=replace(
            base.surrogate,
            k=2,
            index_driver=NeighborIndex.BPANN_DISK,
            enn_storage="disk",
            work_dir=tmp_path,
        ),
        trust_region=RegionalSearchPolicy(count=2, sharing=sharing),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        seed=7,
    )
    for _ in range(3):
        state, batch = baux.ask(state, 2)
        state, result = baux.tell(state, batch.points, -np.sum(batch.points**2, axis=1))
        assert np.all(np.isfinite(result))
    assert (tmp_path / "regions/slot-0-generation-0/num_obs.bin").exists()
    assert (tmp_path / "regions/slot-1-generation-0/num_obs.bin").exists()


def test_gpregions():
    pytest.importorskip("gpytorch")
    cfg = turbo_one(
        num_init=2,
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        seed=9,
    )
    for _ in range(2):
        state, batch = baux.ask(state, 2)
        state, estimate = baux.tell(
            state, batch.points, -np.sum(batch.points**2, axis=1)
        )
        assert estimate.shape == (2,)
        assert np.all(np.isfinite(estimate))


def test_explicitimport():
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        seed=7,
    )
    state, warmup = baux.ask(state, 1)
    state, _ = baux.tell(state, warmup.points, np.array([0.0]))
    state, _ = baux.import_observations(state, np.array([[0.5, 0.5]]), np.array([1.0]))
    state, pending = baux.ask(state, 2)
    state, _ = baux.import_observations(state, np.array([[0.6, 0.6]]), np.array([2.0]))
    state, _ = baux.tell(state, pending.points, np.array([3.0, 4.0]))
    assert state.num_observations == 5
    with pytest.raises(ValueError, match="explicit import"):
        baux.tell(state, pending.points, np.array([3.0, 4.0]))
    assert state.num_observations == 5


def test_001():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    state = baux.init(
        domain=bounds,
        config=turbo_zero(),
        seed=0,
    )
    state, batch = baux.ask(state, 4)
    x = batch.points
    y = -np.sum(x**2, axis=1)

    assert not x.flags.writeable
    assert y.flags.writeable

    state, mu = baux.tell(state, x, y)

    # Verify that the arrays were marked read-only
    assert not x.flags.writeable
    assert not mu.flags.writeable

    with pytest.raises(ValueError):
        x[0, 0] = 999.0
    with pytest.raises(ValueError):
        mu[0] = 999.0
