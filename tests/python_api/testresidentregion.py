import numpy as np
import pytest

from baux.experimental import MetalParamBlock, MetalRegionOptimizer, MetalWeights


def test_residentloop():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        capacity=3,
    )
    first, second = opt.ask(12), opt.ask(13)
    assert (first.region, second.region) == (0, 1)
    view = first.weights()
    assert view.size == 17
    with pytest.raises(BufferError, match="release"):
        opt.tell(first, 2.0)
    del view
    with pytest.raises(ValueError):
        opt.tell(first, float("nan"))
    assert opt.tell(second, 3.0)
    assert opt.tell(first, 2.0)
    assert opt.best == (1, 3.0)
    assert opt.trials == [1, 1]
    with pytest.raises(ValueError, match="no longer pending"):
        opt.tell(first, 2.0)
    opt.restart(0, MetalWeights(starts[0]), 4.0)
    fresh = opt.ask(14, region=0)
    assert fresh.generation == 1
    assert opt.tell(fresh, 5.0)
    assert opt.best == (0, 5.0)
    assert opt.resident_bytes > 0


def test_nearestsharing():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        capacity=3,
        sharing="nearest_center",
    )
    first = opt.ask(12, region=0)
    second = opt.ask(13, region=1)
    assert opt.tell(second, 2.0)
    assert opt.sharing_backlog == [1, 0]
    with pytest.raises(ValueError, match="pending"):
        opt.flush_sharing(0)
    opt.tell(first, -1.0)
    opt.flush_sharing(0)
    assert opt.sharing_backlog == [0, 0]
    assert opt.delivered_imports == 1
    assert opt.trials == [1, 1]
    assert opt.best == (1, 2.0)


def test_sharedzero():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.0)],
        sharing="shared",
    )
    proposal = opt.ask(8, region=1)
    opt.tell(proposal, 2.0)
    assert opt.sharing_backlog == [1, 0]
    opt.flush_sharing(0)
    assert opt.delivered_imports == 1


def test_cancelproposal():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        capacity=3,
    )
    proposal = opt.ask(12, region=0)
    view = proposal.weights()
    with pytest.raises(BufferError, match="release"):
        opt.cancel(proposal)
    del view
    opt.cancel(proposal)
    assert opt.trials == [0, 0]
    with pytest.raises(ValueError, match="no longer pending"):
        opt.tell(proposal, 2.0)
    fresh = opt.ask(13, region=0)
    assert fresh.generation == proposal.generation
    assert opt.tell(fresh, 2.0)
    assert opt.trials == [1, 0]


def test_externalimport():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        capacity=3,
    )
    foreign = MetalWeights(np.full(17, 0x4000, dtype=np.uint16))
    opt.import_region(0, foreign, 2.0)
    assert opt.delivered_imports == 1
    assert opt.trials == [0, 0]
    pending = opt.ask(12, region=0)
    opt.import_region(0, foreign, 3.0)
    assert opt.sharing_backlog == [1, 0]
    opt.tell(pending, 1.5)
    opt.flush_sharing(0)
    assert opt.delivered_imports == 2
    assert opt.trials == [1, 0]


@pytest.mark.parametrize(
    "sharing,expected",
    [("nearest_center", [0]), ("shared", [0, 1])],
)
def test_policyimport(sharing, expected):
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        capacity=3,
        sharing=sharing,
    )
    foreign = MetalWeights(starts[0])
    assert opt.import_observations(foreign, 2.0) == expected
    assert opt.sharing_backlog == [int(slot in expected) for slot in range(2)]
    for slot in expected:
        opt.flush_sharing(slot)
    assert opt.delivered_imports == len(expected)
    assert opt.trials == [0, 0]


def test_independentimport():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = MetalRegionOptimizer(
        starts,
        np.array([0, 1], dtype=np.float32),
        [MetalParamBlock(1, 0, 17, 0.25)],
        sharing="independent",
    )
    foreign = MetalWeights(starts[0])
    with pytest.raises(ValueError, match="explicit import_region"):
        opt.import_observations(foreign, 2.0)
