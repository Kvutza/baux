from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
from baux import Telemetry, create_optimizer, turbo_enn, turbo_one, turbo_zero
from baux.bo import AskBatch, AskId, Optimizer
from baux.turbo.config import (
    NeighborIndex,
    RandomAcquisitionConfig,
    RegionalSearchPolicy,
    RestartCenterPolicy,
    TrustRegionBounds,
    UCBAcquisitionConfig,
    lhd_only,
)


def test_factorysignature():
    assert set(inspect.signature(create_optimizer).parameters) == {
        "bounds",
        "config",
        "rng",
    }


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
    opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(42))
    x = opt.ask(2)

    assert isinstance(opt, Optimizer)
    assert x.shape == (2, 2)
    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert isinstance(opt.telemetry(), Telemetry)


def test_surrogate():
    bounds = np.array([[-2.0, 3.0], [10.0, 20.0]])
    enn = create_optimizer(
        bounds=bounds,
        config=turbo_enn(num_init=2),
        rng=np.random.default_rng(42),
    )
    zero = create_optimizer(
        bounds=bounds,
        config=turbo_zero(num_init=2),
        rng=np.random.default_rng(42),
    )

    assert enn._inner.has_surrogate()
    assert not zero._inner.has_surrogate()
    x = enn.ask(2)
    estimate = enn.tell(x, -np.sum(x**2, axis=1))
    assert estimate.shape == (2,)
    assert np.all(np.isfinite(estimate))
    assert np.allclose(np.asarray(enn._xobs.view()), x)


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
    opt = create_optimizer(
        bounds=np.array([[-2.0, 3.0], [10.0, 20.0]]),
        config=cfg,
        rng=np.random.default_rng(42),
    )
    assert opt.num_regions == 3
    assert opt.tr_length == pytest.approx(0.3)
    for iteration in range(4):
        x = opt.ask(3)
        if iteration > 0:
            with pytest.raises(ValueError, match="outstanding"):
                opt.ask(1)
        y = -np.sum(x**2, axis=1)
        estimate = opt.tell(x, y)
        assert estimate.shape == (3,)
        assert np.all(np.isfinite(estimate))


def test_partialcompletion():
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        rng=np.random.default_rng(31),
    )
    opt.ask(1)
    warmup_ids = opt.pending_ids
    assert warmup_ids is not None and len(warmup_ids) == 1
    opt.complete_ids(warmup_ids, np.array([0.0]))
    opt.ask(3)
    ids = opt.pending_ids
    assert ids is not None and len(ids) == 3
    opt.complete_ids(ids[2:], np.array([3.0]))
    assert opt.pending_ids == ids[:2]
    with pytest.raises(ValueError, match="duplicate"):
        opt.complete_ids(ids[2:], np.array([3.0]))
    opt.complete_ids(ids[:2], np.array([1.0, 2.0]))
    assert opt.pending_ids is None
    assert opt.region_count == 4
    assert opt.ask(3).shape == (3, 2)
    next_ids = opt.pending_ids
    assert next_ids is not None and len(next_ids) == 3
    opt.cancel_ids(next_ids[:2])
    opt.complete_ids(next_ids[2:], np.array([4.0]))
    assert opt.pending_ids is None
    assert opt.region_count == 5


def test_typedasklifecycle():
    from baux import AskBatch as PublicAskBatch
    from baux import AskId as PublicAskId

    assert PublicAskBatch is AskBatch
    assert PublicAskId is AskId
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        rng=np.random.default_rng(31),
    )
    batch = opt.ask_batch(1)
    assert isinstance(batch, AskBatch)
    assert len(batch.ids) == 1 and isinstance(batch.ids[0], AskId)
    assert batch.points.shape == (1, 2)
    opt.complete(batch, np.array([0.0]))
    assert opt.pending_ids is None
    next_batch = opt.ask_batch(2)
    opt.cancel([next_batch.ids[0]])
    opt.complete([next_batch.ids[1]], np.array([1.0]))
    assert opt.pending_ids is None


def assertobservations(opt, points, values):
    observed = {
        tuple(point): float(value)
        for point, value in zip(
            np.asarray(opt._xobs.view()),
            np.asarray(opt._yobs.view()).reshape(-1),
            strict=True,
        )
    }
    for point, value in zip(points, values, strict=True):
        assert observed[tuple(point)] == value


def assertticketvalues(opt, tickets, values):
    points = tuple(ticket.points[0] for ticket in tickets)
    assertobservations(opt, points, values)


def test_overlapaskids():
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        rng=np.random.default_rng(32),
    )
    warmup = opt.ask_batch(1)
    opt.complete(warmup, np.array([0.0]))

    first = opt.ask_batch(1)
    second = opt.ask_batch(1)
    assert set(opt.pending_ids) == {first.ids[0].value, second.ids[0].value}
    opt.complete(first, np.array([10.0]))
    assert opt.pending_ids == (second.ids[0].value,)

    replacement = opt.ask_batch(1)
    assert replacement.ids[0] not in (*first.ids, *second.ids)
    assert set(opt.pending_ids) == {second.ids[0].value, replacement.ids[0].value}
    opt.complete(second, np.array([20.0]))
    opt.complete(replacement, np.array([30.0]))
    assert opt.pending_ids is None
    assertticketvalues(opt, (first, second, replacement), (10.0, 20.0, 30.0))


def testregionrestart():
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=3,
            allocation="least_trials",
            success_tolerance=1,
        ),
    )
    opt = create_optimizer(
        bounds=np.array([[10.0, 20.0], [-2.0, 2.0]]),
        config=config,
        rng=np.random.default_rng(93),
    )
    warmup = opt.ask_batch(1)
    opt.complete(warmup, np.array([0.0]))
    group = opt.ask_batch(3)
    opt.complete([group.ids[0]], np.array([10.0]))

    center = np.array([15.0, 0.0])
    abandoned = opt.restart_region(1, center)
    assert center.flags.writeable
    assert abandoned == (group.ids[1],)
    assert opt.pending_ids == (group.ids[2].value,)
    with pytest.raises(ValueError, match="duplicate"):
        opt.complete(abandoned, np.array([20.0]))

    count = opt.region_count
    opt.import_observations(group.points[1:2], np.array([20.0]))
    assert opt.region_count == count + 1
    assert opt.pending_ids == (group.ids[2].value,)
    opt.complete([group.ids[2]], np.array([30.0]))
    assert opt.pending_ids is None
    assertobservations(opt, group.points, (10.0, 20.0, 30.0))


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
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=config,
        rng=np.random.default_rng(83),
    )
    assert opt.num_regions == 2

    joint_config = replace(
        turbo_enn(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=2,
            allocation="joint_acquisition",
        ),
    )
    joint_opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=joint_config,
        rng=np.random.default_rng(85),
    )
    assert joint_opt.ask(1).shape == (1, 1)

    random_config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(
            count=2,
            acquisitions=(None, RandomAcquisitionConfig()),
        ),
    )
    random_opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=random_config,
        rng=np.random.default_rng(84),
    )
    assert random_opt.ask(2).shape == (2, 1)


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
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        rng=np.random.default_rng(7),
    )
    for _ in range(3):
        x = opt.ask(2)
        result = opt.tell(x, -np.sum(x**2, axis=1))
        assert np.all(np.isfinite(result))
    assert (tmp_path / "regions/slot-0-generation-0/num_obs.bin").exists()
    assert (tmp_path / "regions/slot-1-generation-0/num_obs.bin").exists()


def test_gpregions():
    cfg = turbo_one(
        num_init=2,
        trust_region=RegionalSearchPolicy(count=2),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        rng=np.random.default_rng(9),
    )
    for _ in range(2):
        x = opt.ask(2)
        estimate = opt.tell(x, -np.sum(x**2, axis=1))
        assert estimate.shape == (2,)
        assert np.all(np.isfinite(estimate))


def test_explicitimport():
    cfg = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=cfg,
        rng=np.random.default_rng(7),
    )
    warmup = opt.ask(1)
    opt.tell(warmup, np.array([0.0]))
    opt.import_observations(np.array([[0.5, 0.5]]), np.array([1.0]))
    pending = opt.ask(2)
    opt.import_observations(np.array([[0.6, 0.6]]), np.array([2.0]))
    opt.tell(pending, np.array([3.0, 4.0]))
    assert opt.region_count == 5
    with pytest.raises(ValueError, match="explicit import"):
        opt.tell(pending, np.array([3.0, 4.0]))
    assert opt.region_count == 5


def test_001():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_zero(),
        rng=np.random.default_rng(0),
    )
    x = opt.ask(4)
    y = -np.sum(x**2, axis=1)

    assert x.flags.writeable
    assert y.flags.writeable

    opt.tell(x, y)

    # Verify that the arrays were marked read-only
    assert not x.flags.writeable
    assert not y.flags.writeable

    with pytest.raises(ValueError):
        x[0, 0] = 999.0
    with pytest.raises(ValueError):
        y[0] = 999.0
