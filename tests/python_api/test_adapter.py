from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from baux import _rust, create_optimizer, turbo_one
from baux.turbo.config import (
    ExternalSurrogate,
    RegionalSearchPolicy,
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


def _make(gp: StubGP):
    bounds = np.array([[-2.0, 2.0], [10.0, 20.0]])
    opt = _rust.create_optimizer(
        bounds,
        "gp",
        10,
        1,
        4,
        42,
        cfg={
            "acquisition": "ucb",
            "min_candidates": 8,
            "max_candidates": 8,
            "num_candidates_factor": 1.0,
        },
        gp=gp,
        fit_steps=7,
    )
    return bounds, opt


def test_gpcallsarebatched():
    gp = StubGP()
    bounds, opt = _make(gp)
    x, opt = opt.ask(1, seed=101)
    y = np.array([[1.25]])
    yvar = np.array([[0.05]])
    opt = opt.tell(x, y, seed=102, y_var=yvar)

    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert len(gp.fits) == 1
    fit_x, fit_y, fit_var = gp.fits[0]
    assert np.all((fit_x >= 0.0) & (fit_x <= 1.0))
    np.testing.assert_array_equal(fit_y, y)
    np.testing.assert_array_equal(fit_var, yvar)

    x2, opt = opt.ask(2, seed=103)
    assert x2.shape == (2, 2)
    assert len(gp.preds) == 1
    assert gp.preds[0].shape == (8, 2)


def test_yvarerrorisatomic():
    gp = StubGP()
    _, opt = _make(gp)
    x, opt = opt.ask(1, seed=201)
    opt = opt.tell(x, np.array([[1.0]]), seed=202, y_var=np.array([[0.1]]))

    with pytest.raises(ValueError, match="y_var must be provided"):
        opt.tell(x, np.array([[2.0]]), seed=203)

    assert opt.num_observations() == 1
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
    _, opt = _make(gp)
    x, opt = opt.ask(1, seed=301)

    assert opt.init_progress() == (0, 1)
    with pytest.raises(ValueError, match="fit failed"):
        opt.tell(x, np.array([[1.0]]), seed=302)

    assert opt.num_observations() == 0
    # Pure immutability on error: opt is completely unchanged on failure
    assert opt.init_progress() == (0, 1)
    opt = opt.tell(x, np.array([[1.0]]), seed=303)
    assert opt.num_observations() == 1
    assert opt.init_progress() is None


def test_forkisolation():
    gp = StubGP()
    _bounds, opt = _make(gp)
    x, opt = opt.ask(1, seed=501)
    opt = opt.tell(x, np.array([[1.0]]), seed=502)
    assert opt.num_observations() == 1

    opt_fork = opt.fork()
    assert opt_fork.digest_hex() == opt.digest_hex()
    assert opt_fork.num_observations() == 1

    # Branch A advances independently
    xa, opt_a = opt.ask(1, seed=503)
    opt_a = opt_a.tell(xa, np.array([[2.0]]), seed=504)

    # Branch B advances independently from fork snapshot
    xb, opt_b = opt_fork.ask(1, seed=503)
    opt_b = opt_b.tell(xb, np.array([[5.0]]), seed=504)

    # Deterministic generation with same seed
    np.testing.assert_allclose(xa, xb)

    # Pure fork isolation: branches diverge and original snapshots remain immutable
    assert opt.num_observations() == 1
    assert opt_fork.num_observations() == 1
    assert opt_a.num_observations() == 2
    assert opt_b.num_observations() == 2
    assert opt_a.digest_hex() != opt_b.digest_hex()
    assert opt.digest_hex() == opt_fork.digest_hex()


def test_001():
    class BadPredict(StubGP):
        def predict(self, x):
            return SimpleNamespace(mu=np.zeros((1, 1)), sigma=np.ones((1, 1)))

    gp = BadPredict()
    _, opt = _make(gp)
    x, opt = opt.ask(1, seed=401)
    opt = opt.tell(x, np.array([[1.0]]), seed=402)

    with pytest.raises(ValueError, match="predict must return"):
        opt.ask(2, seed=403)


@pytest.mark.parametrize(
    "sharing",
    ["box_contain", "ellipsoid_ball", "nearest_center", "origin_only"],
)
def test_regionalprovider(sharing):
    gp = StubGP()
    local_models: list[StubGP] = []

    def factory(slot: int, generation: int) -> StubGP:
        assert generation == 0
        model = StubGP()
        local_models.append(model)
        return model

    opt = _rust.create_optimizer(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        "gp",
        num_init=1,
        num_regions=3,
        gp=gp,
        gp_factory=factory,
        fit_steps=7,
        cfg={
            "region_sharing": sharing,
            "acquisition": "ucb",
            "min_candidates": 8,
            "max_candidates": 8,
        },
    )
    assert opt.num_regions() == 3
    for _ in range(3):
        x, opt = opt.ask(3)
        opt = opt.tell(x, -np.sum(x**2, axis=1, keepdims=True))
    assert len(gp.fits) == 3
    assert not gp.preds
    assert len(local_models) == 3
    assert all(model.fits and model.preds for model in local_models)


def test_missingfactory():
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
    opt = _rust.create_optimizer(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        "gp",
        num_init=1,
        num_regions=2,
        gp=global_model,
        gp_factory=lambda slot, generation: StubGP(),
        fit_steps=7,
        cfg={"region_sharing": "origin_only", "min_candidates": 8, "max_candidates": 8},
    )
    warmup, opt = opt.ask(1)
    opt = opt.tell(warmup, np.array([[1.0]]))
    proposal, opt = opt.ask(2)
    outcomes = np.ones((2, 1))
    with pytest.raises(ValueError, match="regional global fit failed"):
        opt.tell(proposal, outcomes)
    # Under pure functional contract, failed tell leaves opt completely unchanged
    assert opt.num_observations() == 1
    assert opt.ingestion_failure is None
    assert len(global_model.fits) == 1


def test_regionmodels():
    global_model = StubGP()
    local_models: list[StubGP] = []

    def factory(slot: int, generation: int) -> StubGP:
        assert slot == len(local_models)
        assert generation == 0
        model = StubGP()
        local_models.append(model)
        return model

    opt = _rust.create_optimizer(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        "gp",
        num_init=1,
        num_regions=3,
        gp=global_model,
        gp_factory=factory,
        fit_steps=7,
        cfg={
            "region_sharing": "origin_only",
            "acquisition": "ucb",
            "min_candidates": 8,
            "max_candidates": 8,
        },
    )
    assert len(local_models) == 3
    x, opt = opt.ask(1)
    opt = opt.tell(x, np.array([[0.1]]))
    x, opt = opt.ask(3)
    opt = opt.tell(x, np.array([[0.2], [0.3], [0.4]]))

    assert global_model.fits[-1][0].shape == (4, 2)
    assert [model.fits[-1][0].shape[0] for model in local_models] == [2, 2, 2]
    assert all(model.preds for model in local_models)


def test_reusedprovider():
    model = StubGP()
    for global_model in [model, StubGP()]:
        with pytest.raises(ValueError, match="reused a provider"):
            _rust.create_optimizer(
                np.array([[0.0, 1.0]]),
                "gp",
                num_init=1,
                num_regions=2,
                gp=global_model,
                gp_factory=lambda slot, generation: model,
                fit_steps=7,
                cfg={"region_sharing": "origin_only"},
            )


def test_external():
    providers: list[StubGP] = []

    def factory() -> StubGP:
        model = StubGP()
        providers.append(model)
        return model

    config = replace(
        turbo_one(num_init=1),
        surrogate=ExternalSurrogate(factory, fit_steps=7),
        trust_region=RegionalSearchPolicy(count=3, sharing="origin_only"),
    )
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        rng=np.random.default_rng(11),
    )
    x = opt.ask(1)
    opt.tell(x, np.array([[0.1]]))
    x = opt.ask(3)
    opt.tell(x, np.array([[0.2], [0.3], [0.4]]))

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
