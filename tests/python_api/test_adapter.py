from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import baux
import numpy as np
import pytest
from baux import _rust
from baux.turbo.config import (
    CandidateConfig,
    ExternalSurrogate,
    RegionalSearchPolicy,
    UCBAcquisitionConfig,
    turbo_one,
    turbo_zero,
)


class StubGP:
    def __init__(self) -> None:
        self.fits: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        self.preds: list[np.ndarray] = []

    def fit(self, x, y, yvar, *, steps):
        assert steps == 7
        self.fits.append(
            (
                np.asarray(x).copy(),
                np.asarray(y).copy(),
                None if yvar is None else np.asarray(yvar).copy(),
            )
        )
        return SimpleNamespace(lengthscales=np.ones(x.shape[1]))

    def predict(self, x):
        x = np.asarray(x)
        self.preds.append(x.copy())
        return SimpleNamespace(
            mu=x.sum(1, keepdims=True),
            sigma=np.full((len(x), 1), 0.1),
        )

    def draw(self, x, n, seed):
        return np.random.default_rng(seed).normal(size=(n, len(x), 1))


def _makestate(gp: StubGP, num_init: int = 1):
    bounds = np.array([[-2.0, 2.0], [10.0, 20.0]])
    config = replace(
        turbo_one(num_init=num_init),
        surrogate=ExternalSurrogate(lambda: gp, fit_steps=7),
        acquisition=UCBAcquisitionConfig(),
        candidates=CandidateConfig(num_candidates=8),
    )
    return bounds, baux.init(domain=bounds, config=config, seed=42)


def test_gpcallsarebatched():
    gp = StubGP()
    bounds, state = _makestate(gp)
    state, b1 = baux.ask(state, 1)
    x = b1.points
    y = np.array([[1.25]])
    yvar = np.array([[0.05]])
    state, _ = baux.tell(state, x, y, y_var=yvar)

    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert len(gp.fits) == 1
    fit_x, fit_y, fit_var = gp.fits[0]
    assert np.all((fit_x >= 0.0) & (fit_x <= 1.0))
    np.testing.assert_array_equal(fit_y, y)
    np.testing.assert_array_equal(fit_var, yvar)

    state, b2 = baux.ask(state, 2)
    x2 = b2.points
    assert x2.shape == (2, 2)
    assert len(gp.preds) == 2
    assert gp.preds[-1].shape == (8, 2)


def test_yvarerrorisatomic():
    gp = StubGP()
    _, state = _makestate(gp)
    state, b1 = baux.ask(state, 1)
    state, _ = baux.tell(state, b1, np.array([[1.0]]), y_var=np.array([[0.1]]))

    with pytest.raises(ValueError, match="y_var must be provided"):
        baux.tell(state, b1, np.array([[2.0]]))

    assert state.num_observations == 1
    assert len(gp.fits) == 1


def test_fiterrorisatomic():
    class FailOnce(StubGP):
        fail = True

        def fit(self, x, y, yvar, *, steps):
            if self.fail:
                self.fail = False
                raise RuntimeError("fit failed")
            return super().fit(x, y, yvar, steps=steps)

    gp = FailOnce()
    _, state = _makestate(gp)
    s1, b1 = baux.ask(state, 1)

    assert s1.init_progress == (0, 1)
    with pytest.raises(ValueError, match="fit failed"):
        baux.tell(s1, b1, np.array([[1.0]]))

    assert s1.num_observations == 0
    # Pure immutability on error: s1 is completely unchanged on failure
    assert s1.init_progress == (0, 1)
    s2, _ = baux.tell(s1, b1, np.array([[1.0]]))
    assert s2.num_observations == 1
    assert s2.init_progress is None


def test_forkisolation():
    gp = StubGP()
    bounds = np.array([[-2.0, 2.0], [10.0, 20.0]])
    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(lambda: gp, fit_steps=7),
    )
    state = baux.init(domain=bounds, config=config, seed=501)
    state, b1 = baux.ask(state, 1)
    state, _ = baux.tell(state, b1, np.array([[1.0]]))
    assert state.num_observations == 1

    state_fork = state.fork()
    assert state_fork.digest() == state.digest()
    assert state_fork.num_observations == 1

    # Branch A advances independently
    state_a, ba = baux.ask(state.fork(seed=503), 1)
    state_a, _ = baux.tell(state_a, ba, np.array([[2.0]]))

    # Branch B advances independently from fork snapshot
    state_b, bb = baux.ask(state_fork.fork(seed=503), 1)
    state_b, _ = baux.tell(state_b, bb, np.array([[5.0]]))

    # Deterministic generation with same seed
    np.testing.assert_allclose(ba.points, bb.points)

    # Pure fork isolation: branches diverge and original snapshots remain immutable
    assert state.num_observations == 1
    assert state_fork.num_observations == 1
    assert state_a.num_observations == 2
    assert state_b.num_observations == 2
    assert state_a.digest() != state_b.digest()
    assert state.digest() == state_fork.digest()


def test_001():
    class BadPredict(StubGP):
        def predict(self, x):
            return SimpleNamespace(mu=np.zeros((1, 1)), sigma=np.ones((1, 1)))

    gp = BadPredict()
    _, state = _makestate(gp)
    state, b1 = baux.ask(state, 1)
    state, _ = baux.tell(state, b1, np.array([[1.0]]))

    with pytest.raises(ValueError, match="predict must return"):
        baux.ask(state, 2)


@pytest.mark.parametrize(
    "sharing",
    ["box_contain", "ellipsoid_ball", "nearest_center", "origin_only"],
)
def test_regionalprovider(sharing):
    all_models: list[StubGP] = []

    def factory() -> StubGP:
        model = StubGP()
        all_models.append(model)
        return model

    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(factory, fit_steps=7),
        trust_region=RegionalSearchPolicy(
            count=3,
            sharing=sharing,
            acquisitions=(
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
            ),
        ),
        candidates=CandidateConfig(num_candidates=8),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=42,
    )
    assert state.num_regions == 3
    for _ in range(3):
        state, batch = baux.ask(state, 3)
        state, _ = baux.tell(
            state, batch, -np.sum(batch.points**2, axis=1, keepdims=True)
        )

    assert len(all_models) == 4
    global_m = all_models[0]
    regional_m = all_models[1:]
    assert len(global_m.fits) == 3
    assert len(regional_m) == 3
    assert all(model.fits and model.preds for model in regional_m)


def test_missingfactory():
    with pytest.raises(TypeError, match="surrogate factory must be callable"):
        ExternalSurrogate(factory=None)
    with pytest.raises(ValueError, match="requires a regional surrogate factory"):
        _rust.create_optimizer(
            np.array([[0.0, 1.0]]),
            "gp",
            num_init=1,
            num_regions=2,
            gp=StubGP(),
            fit_steps=7,
            cfg={"region_sharing": "origin_only"},
        )


def testbadpolicybinding():
    with pytest.raises(ValueError, match="unknown regional allocation policy"):
        _rust.create_optimizer(
            np.array([[0.0, 1.0]]),
            "zero",
            10,
            1,
            2,
            42,
            cfg={"region_sharing": "box_contain", "region_allocation": "weighted"},
        )
    with pytest.raises(ValueError, match="unknown regional restart memory policy"):
        _rust.create_optimizer(
            np.array([[0.0, 1.0]]),
            "zero",
            10,
            1,
            2,
            42,
            cfg={"region_sharing": "box_contain", "region_restart_memory": "all"},
        )
    with pytest.raises(ValueError, match="unknown regional controller"):
        _rust.create_optimizer(
            np.array([[0.0, 1.0]]),
            "zero",
            10,
            1,
            2,
            42,
            cfg={"region_sharing": "box_contain", "region_controller": "adaptive"},
        )
    with pytest.raises(ValueError, match="unknown regional acquisition"):
        _rust.create_optimizer(
            np.array([[0.0, 1.0]]),
            "zero",
            10,
            1,
            2,
            42,
            cfg={
                "region_sharing": "box_contain",
                "region_acquisitions": ["ucb", "exploit"],
            },
        )


def test_failedingestion():
    class FailAfterWarmup(StubGP):
        def fit(self, x, y, yvar, *, steps):
            if self.fits:
                raise RuntimeError("regional global fit failed")
            return super().fit(x, y, yvar, steps=steps)

    global_model = FailAfterWarmup()
    first = True

    def factory() -> StubGP:
        nonlocal first
        if first:
            first = False
            return global_model
        return StubGP()

    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(factory, fit_steps=7),
        trust_region=RegionalSearchPolicy(
            count=2,
            sharing="origin_only",
            acquisitions=(UCBAcquisitionConfig(), UCBAcquisitionConfig()),
        ),
        candidates=CandidateConfig(num_candidates=8),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=42,
    )
    state, warmup = baux.ask(state, 1)
    state, _ = baux.tell(state, warmup, np.array([[1.0]]))
    state, proposal = baux.ask(state, 2)
    outcomes = np.ones((2, 1))
    with pytest.raises(ValueError, match="regional global fit failed"):
        baux.tell(state, proposal, outcomes)
    # Under pure functional contract, failed tell leaves state completely unchanged
    assert state.num_observations == 1
    assert state.ingestion_failure is None
    assert len(global_model.fits) == 1


def test_regionmodels():
    all_models: list[StubGP] = []

    def factory() -> StubGP:
        model = StubGP()
        all_models.append(model)
        return model

    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(factory, fit_steps=7),
        trust_region=RegionalSearchPolicy(
            count=3,
            sharing="origin_only",
            acquisitions=(
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
            ),
        ),
        candidates=CandidateConfig(num_candidates=8),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=42,
    )
    global_model = all_models[0]
    local_models = all_models[1:]
    assert len(local_models) == 3

    state, b1 = baux.ask(state, 1)
    state, _ = baux.tell(state, b1, np.array([[0.1]]))
    state, b2 = baux.ask(state, 3)
    state, _ = baux.tell(state, b2, np.array([[0.2], [0.3], [0.4]]))

    assert global_model.fits[-1][0].shape == (4, 2)
    assert [model.fits[-1][0].shape[0] for model in local_models] == [2, 2, 2]
    assert all(model.preds for model in local_models)


def test_reusedprovider():
    model = StubGP()
    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(lambda: model, fit_steps=7),
        trust_region=RegionalSearchPolicy(
            count=2,
            sharing="origin_only",
            acquisitions=(UCBAcquisitionConfig(), UCBAcquisitionConfig()),
        ),
    )
    with pytest.raises(ValueError, match="reused a provider"):
        baux.init(np.array([[0.0, 1.0]]), config=config)


def test_external():
    providers: list[StubGP] = []

    def factory() -> StubGP:
        model = StubGP()
        providers.append(model)
        return model

    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(factory, fit_steps=7),
        trust_region=RegionalSearchPolicy(
            count=3,
            sharing="origin_only",
            acquisitions=(
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
                UCBAcquisitionConfig(),
            ),
        ),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=11,
    )
    state, b1 = baux.ask(state, 1)
    state, _ = baux.tell(state, b1, np.array([[0.1]]))
    state, b2 = baux.ask(state, 3)
    state, _ = baux.tell(state, b2, np.array([[0.2], [0.3], [0.4]]))

    assert len(providers) == 4
    assert providers[0].fits[-1][0].shape == (4, 2)
    assert [provider.fits[-1][0].shape[0] for provider in providers[1:]] == [2, 2, 2]


def testtrialretry():
    from baux._trials import Loop, Space

    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    loop = Loop(Space({"x": (0.0, 1.0)}), config, 37)
    initial = {"seed": ({"x": 0.5}, 1.0, None)}
    loop.update(initial)
    warmup = loop.suggest([])
    history = {**initial, "warmup": (warmup, 0.0, None)}
    loop.update(history)
    first = loop.suggest([])
    first_ids = loop.optimizer.pending_ids
    assert first_ids is not None
    second = loop.suggest([first])
    second_ids = loop.optimizer.pending_ids
    assert second_ids is not None and len(second_ids) == 2
    assert set(second_ids) >= set(first_ids)
    second_id = next(identity for identity in second_ids if identity not in first_ids)

    # Completing one ticket frees its slot while preserving the other ID.
    history = {**history, "first": (first, 2.0, None)}
    loop.update(history)
    assert loop.optimizer.pending_ids == (second_id,)
    third = loop.suggest([second])
    third_ids = loop.optimizer.pending_ids
    assert third_ids is not None and len(third_ids) == 2
    assert second_id in third_ids
    third_id = next(identity for identity in third_ids if identity != second_id)

    history = {**history, "second": (second, 3.0, None)}
    loop.update(history)
    assert loop.optimizer.pending_ids == (third_id,)
    loop.update(
        {
            **history,
            "third": (third, 4.0, None),
        }
    )
    assert loop.optimizer.pending_ids is None
    assert set(loop._seen) == {"seed", "warmup", "first", "second", "third"}


def testoneregionimport():
    from baux._trials import Loop, Space

    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=1),
    )
    loop = Loop(Space({"x": (0.0, 1.0)}), config, 41)
    initial = {"seed": ({"x": 0.5}, 1.0, None)}
    loop.update(initial)
    proposal = loop.suggest([])
    assert loop.optimizer.pending_ids is not None
    loop.update({**initial, "trial": (proposal, 2.0, None)})
    assert loop.optimizer.pending_ids is None
    assert set(loop._seen) == {"seed", "trial"}


def test_completerestart():
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    state = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=77,
    )
    state, batch = baux.ask(state, 2)
    assert len(batch.ids) == 2
    id1, id2 = batch.ids

    # complete
    s1 = baux.complete(state, [id1], np.array([[1.5]]))
    assert s1.pending_ids == (id2.value,)
    assert state.pending_ids == (id1.value, id2.value)  # immutability

    # cancel
    s2 = baux.cancel(s1, [id2])
    assert s2.pending_ids is None

    # restart_region
    s_ask, _ = baux.ask(s2, 1)
    s_rst, abandoned = baux.restart_region(s_ask, 0, np.array([0.5, 0.5]))
    assert isinstance(abandoned, tuple)
    assert isinstance(s_rst, baux.OptimizerState)
