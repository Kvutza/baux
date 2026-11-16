from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _loadmodule():
    import baux
    import baux.turbo.config as _tc

    had_acq = "AcquisitionKind" in baux.__dict__
    had_turbo = "turbo_enn" in baux.__dict__
    had_opt = "create_optimizer" in baux.__dict__

    try:
        for name in ("AcquisitionKind", "turbo_enn"):
            if not hasattr(baux, name) and hasattr(_tc, name):
                setattr(baux, name, getattr(_tc, name))

        def _makeopt(*, bounds, config, rng=None):
            class _FunctionalOptimizer:
                def __init__(self, bounds, config, rng):
                    seed = (
                        int(rng.integers(0, 2**31))
                        if isinstance(rng, np.random.Generator)
                        else (rng if rng is not None else 0)
                    )
                    self.state = baux.init(bounds, config=config, seed=seed)

                def ask(self, num_arms):
                    self.state, batch = baux.ask(self.state, num_arms)
                    return getattr(batch, "points", batch)

                def tell(self, x, y):
                    self.state, _ = baux.tell(self.state, x, y)

                def telemetry(self):
                    return getattr(self.state, "telemetry", None)

            return _FunctionalOptimizer(bounds, config, rng)

        baux.__dict__["create_optimizer"] = _makeopt

        script_path = (
            pathlib.Path(__file__).resolve().parents[1] / "scripts" / "benchdtsel.py"
        )
        spec = importlib.util.spec_from_file_location("benchdtsel", script_path)
        assert spec is not None and spec.loader is not None
        sys.modules.pop("benchdtsel", None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if not had_acq:
            baux.__dict__.pop("AcquisitionKind", None)
        if not had_turbo:
            baux.__dict__.pop("turbo_enn", None)
        if not had_opt:
            baux.__dict__.pop("create_optimizer", None)


def test_001():
    module = _loadmodule()
    bounds = module.make_bounds(3)
    assert bounds.shape == (3, 2)
    assert np.all(bounds[:, 0] == 0.0)
    assert np.all(bounds[:, 1] == 1.0)


def test_002():
    module = _loadmodule()
    x = np.array([[0.2, 0.4], [0.8, 0.1]], dtype=float)
    y = module.synthetic_objective(x)
    assert y.shape == (2, 1)
    assert y.dtype == float


def test_003():
    module = _loadmodule()

    class Recorder:
        def __init__(self):
            self.tell_args = []

        def tell(self, x, y):
            self.tell_args.append((x.copy(), y.copy()))

    rng = np.random.default_rng(0)
    rec = Recorder()
    module.seed_observations(rec, rng, num_obs=5, num_dim=2, batch_size=2)
    assert len(rec.tell_args) == 3
    assert rec.tell_args[0][0].shape == (2, 2)
    assert rec.tell_args[-1][0].shape == (1, 2)


def test_004(capsys):
    module = _loadmodule()
    module.summarize([1.0, 2.0, 3.0], "dt_sel")
    out = capsys.readouterr().out
    assert "dt_sel_ms" in out
    assert "mean=" in out
    assert "p95=" in out


def test_005(monkeypatch, capsys):
    import baux

    module = _loadmodule()

    class FakeTelemetry:
        dt_sel = 0.001
        dt_gen = 0.002
        dt_fit = 0.003

    class FakeState:
        def __init__(self, bounds, config, seed):
            self.bounds = bounds
            self.config = config
            self.seed = seed
            self.telemetry = FakeTelemetry()

    init_calls = []
    ask_calls = []
    tell_calls = []

    def fake_init(domain, config=None, *, seed=None, **kwargs):
        assert domain.shape == (2, 2)
        assert config["num_init"] == 2
        assert seed is not None
        init_calls.append((domain, config, seed))
        return FakeState(domain, config, seed)

    def fake_ask(state, num_arms):
        ask_calls.append(num_arms)
        batch = type("AskBatch", (), {"points": np.zeros((num_arms, 2))})()
        return state, batch

    def fake_tell(state, x, y, **kwargs):
        tell_calls.append((x, y))
        return state, y

    # Force-resolve lazy attributes first before monkeypatching
    _ = (baux.init, baux.ask, baux.tell)

    monkeypatch.setattr(baux, "init", fake_init)
    monkeypatch.setattr(baux, "ask", fake_ask)
    monkeypatch.setattr(baux, "tell", fake_tell)

    class FunctionalOpt:
        def __init__(self, bounds, config, rng):
            self.state = baux.init(bounds, config=config, seed=rng)

        def ask(self, num_arms):
            self.state, batch = baux.ask(self.state, num_arms)
            return batch.points

        def tell(self, x, y):
            self.state, _ = baux.tell(self.state, x, y)

        def telemetry(self):
            return self.state.telemetry

    created = {"n": 0}

    def enn_config4(*, enn, num_init, acquisition):
        created["n"] += 1
        assert created["n"] == 1
        return {"enn": enn, "num_init": num_init, "acquisition": acquisition}

    monkeypatch.setattr(module, "turbo_enn", enn_config4)
    monkeypatch.setattr(
        module,
        "create_optimizer",
        lambda *, bounds, config, rng: FunctionalOpt(bounds, config, rng),
    )

    module.run_benchmark(
        [
            "--timed-asks",
            "1",
            "--warmup-asks",
            "1",
            "--num-obs",
            "3",
            "--num-dim",
            "2",
            "--seed-batch-size",
            "2",
            "--num-arms",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert "setup" in out
    assert "timed_asks=1" in out
    assert len(init_calls) == 1
    assert len(ask_calls) == 2
    assert len(tell_calls) == 2


def test_006():
    module = _loadmodule()
    with pytest.raises(
        ValueError, match="num_obs, num_dim, and k must all be positive"
    ):
        module.run_benchmark(["--num-obs", "0", "--num-dim", "2"])


def test_funcadapter():
    import baux

    class FakeState:
        def __init__(self, bounds, config, seed):
            self.bounds = bounds
            self.config = config
            self.seed = seed
            self.telemetry = "fake_telemetry"

    init_calls = []
    ask_calls = []
    tell_calls = []

    def fake_init(domain, config=None, *, seed=None, **kwargs):
        init_calls.append((domain, config, seed))
        return FakeState(domain, config, seed)

    def fake_ask(state, num_arms):
        ask_calls.append(num_arms)
        return state, SimpleNamespace(points=np.ones((num_arms, 2)))

    def fake_tell(state, x, y, **kwargs):
        tell_calls.append((x, y))
        return state, y

    # Force-resolve lazy attributes via attribute access first
    orig_init = baux.init
    orig_ask = baux.ask
    orig_tell = baux.tell

    had_opt = "create_optimizer" in baux.__dict__
    had_acq = "AcquisitionKind" in baux.__dict__
    had_turbo = "turbo_enn" in baux.__dict__

    baux.__dict__["init"] = fake_init
    baux.__dict__["ask"] = fake_ask
    baux.__dict__["tell"] = fake_tell

    try:
        module = _loadmodule()
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        opt = module.create_optimizer(bounds=bounds, config={"num_init": 2}, rng=123)
        assert len(init_calls) == 1
        pts = opt.ask(2)
        assert pts.shape == (2, 2)
        assert len(ask_calls) == 1
        opt.tell(pts, np.array([0.5, 0.6]))
        assert len(tell_calls) == 1
        assert opt.telemetry() == "fake_telemetry"
    finally:
        baux.__dict__["init"] = orig_init
        baux.__dict__["ask"] = orig_ask
        baux.__dict__["tell"] = orig_tell
        if not had_opt:
            baux.__dict__.pop("create_optimizer", None)
        if not had_acq:
            baux.__dict__.pop("AcquisitionKind", None)
        if not had_turbo:
            baux.__dict__.pop("turbo_enn", None)
