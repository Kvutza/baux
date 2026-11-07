import numpy as np
import pytest

from baux import Device, Portfolio, Region, Span, Var


def test_residentloop():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
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
    opt.restart(0, Var(starts[0]), 4.0)
    fresh = opt.ask(14, region=0)
    assert fresh.generation == 1
    assert opt.tell(fresh, 5.0)
    assert opt.best == (0, 5.0)
    assert opt.resident_bytes > 0


def test_nearestsharing():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
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
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.0)],
        sharing="box_contain",
    )
    proposal = opt.ask(8, region=1)
    opt.tell(proposal, 2.0)
    assert opt.sharing_backlog == [1, 0]
    opt.flush_sharing(0)
    assert opt.delivered_imports == 1


def test_cancelproposal():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
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
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
        capacity=3,
    )
    foreign = Var(np.full(17, 0x4000, dtype=np.uint16))
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
    [("nearest_center", [0]), ("box_contain", [0, 1])],
)
def test_policyimport(sharing, expected):
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
        capacity=3,
        sharing=sharing,
    )
    foreign = Var(starts[0])
    assert opt.import_observations(foreign, 2.0) == expected
    assert opt.sharing_backlog == [int(slot in expected) for slot in range(2)]
    for slot in expected:
        opt.flush_sharing(slot)
    assert opt.delivered_imports == len(expected)
    assert opt.trials == [0, 0]


def test_independentimport():
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    opt = Portfolio(
        starts,
        np.array([0, 1], dtype=np.float32),
        [Span(1, 0, 17, 0.25)],
        sharing="origin_only",
    )
    foreign = Var(starts[0])
    with pytest.raises(ValueError, match="explicit import_region"):
        opt.import_observations(foreign, 2.0)


def test_ellipsoidzero():
    assert Region is not None
    starts = np.full((2, 16), 0x3F80, dtype=np.uint16)
    blocks = [
        Span(1, 0, 8, 0.5, 4.0),
        Span(2, 8, 8, 0.25, 1.0),
    ]
    opt = Portfolio(
        starts,
        np.array([0.0, 0.0], dtype=np.float32),
        blocks,
        capacity=0,
        length_init=0.4,
        sharing="ellipsoid_ball",
    )
    first = opt.ask(11, region=0, acquisition="random")
    accepted = first.weights().read()
    assert opt.tell(first, 5.0)
    assert opt.sharing_backlog == [0, 1]
    opt.flush_sharing(1)
    assert opt.sharing_backlog == [0, 0]
    assert opt.delivered_imports == 1
    adopted = opt.incumbent(1).read()
    np.testing.assert_array_equal(adopted, accepted)
    second = opt.ask(19, region=1, acquisition="random")
    assert opt.tell(second, 6.0)
    assert opt.best == (1, 6.0)


def test_candleapi():
    dev = Device("metal")
    assert dev.is_metal and dev.kind == "metal" and dev == "metal"
    assert dev.memory_info()["name"]
    starts = np.full((2, 17), 0x3E80, dtype=np.uint16)
    values = np.array([0.0, 1.0], dtype=np.float32)
    spans = [Span(1, 0, 17, 0.25)]
    region = Region(Var(starts[0]), 0.0, spans, 2)
    trial = region.ask()
    assert trial.var().read().size == 17
    assert region.tell(trial, 1.0)
    for cap in (0, 4):
        portfolio = Portfolio(starts, values, spans, capacity=cap)
        rt = portfolio.ask(acquisition="random" if cap == 0 else "thompson")
        assert rt.var().read().size == 17
        assert portfolio.tell(rt, 2.0)
