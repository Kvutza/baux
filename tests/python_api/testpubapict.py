from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("baux.baux_rust") is None,
    reason="baux_rust extension unavailable",
)

EXPECTED_74_SYMBOLS: set[str] = {
    "Affine",
    "AskBatch",
    "AskId",
    "Categorical",
    "Device",
    "DistanceMetric",
    "ENNParams",
    "ENNState",
    "Key",
    "Log",
    "NeighborIndex",
    "OptimizerConfig",
    "OptimizerState",
    "Periodic",
    "Portfolio",
    "PyTreeSpec",
    "Region",
    "RegionTrial",
    "RegionalSearchPolicy",
    "Simplex",
    "Space",
    "Stiefel",
    "Telemetry",
    "Tensor",
    "Transform",
    "Trial",
    "TrustRegionBounds",
    "Var",
    "append",
    "ask",
    "calibrate",
    "cancel",
    "capabilities",
    "chain",
    "complete",
    "digest",
    "enn",
    "enn_init",
    "ensure_sync",
    "experimental",
    "fit",
    "fold",
    "freeze",
    "gp",
    "import_observations",
    "init",
    "key",
    "lhd",
    "pareto",
    "persist_index",
    "pmap",
    "posterior",
    "posterior_mu",
    "pytree_bounds",
    "raasp",
    "random_acq",
    "ravel",
    "regions",
    "restart_region",
    "sample",
    "scan",
    "sobol",
    "split",
    "stein",
    "step",
    "tell",
    "thaw",
    "thompson",
    "tr_bounds",
    "ucb",
    "unflatten",
    "uniform_pert",
    "with_geometry",
    "zero",
}

REMOVED_LEGACY_SYMBOLS: list[str] = [
    "Optimizer",
    "create_optimizer",
    "ENN",
    "ENNFitter",
    "TellPredictionError",
    "enn_optimizer",
    "create_zero",
    "create_lhd",
    "enn_append",
    "turbo_one",
    "mlp_spec",
    "quantize_int4",
    "set_path",
    "enn_fit",
    "enn_posterior",
    "enn_sample",
    "quantize_e2m1",
]


class TestPublicAPIExports:
    """Verify the exact 74-symbol pure functional API surface."""

    def test_exactsymbols(self):
        import baux

        assert len(EXPECTED_74_SYMBOLS) == 74
        assert set(baux.__all__) == EXPECTED_74_SYMBOLS
        assert set(baux._LAZY_ATTRS.keys()) == EXPECTED_74_SYMBOLS
        assert set(baux.__all__) == set(baux._LAZY_ATTRS.keys())
        assert len(baux.__all__) == 74

    def test_allimportable(self):
        import baux

        for name in EXPECTED_74_SYMBOLS:
            attr = getattr(baux, name)
            assert attr is not None, f"Attribute {name} resolved to None"

    def test_corecallables(self):
        import baux

        for name in [
            "init",
            "ask",
            "tell",
            "complete",
            "cancel",
            "import_observations",
            "restart_region",
            "enn_init",
            "append",
            "fit",
            "posterior",
            "sample",
            "ensure_sync",
            "persist_index",
            "step",
            "scan",
            "pmap",
            "key",
            "split",
            "fold",
            "digest",
            "freeze",
            "thaw",
            "calibrate",
            "stein",
            "with_geometry",
            "ravel",
            "unflatten",
            "pytree_bounds",
        ]:
            attr = getattr(baux, name)
            assert callable(attr), f"baux.{name} should be callable"

    def test_coreclasses(self):
        import baux

        for name in [
            "Region",
            "Portfolio",
            "Trial",
            "RegionTrial",
            "Device",
            "Tensor",
            "Var",
            "Key",
            "OptimizerState",
            "ENNState",
            "ENNParams",
            "PyTreeSpec",
            "OptimizerConfig",
            "RegionalSearchPolicy",
            "TrustRegionBounds",
            "Transform",
            "AskBatch",
            "AskId",
            "Telemetry",
            "Space",
            "Categorical",
            "Periodic",
            "Simplex",
            "Log",
            "Stiefel",
            "Affine",
            "NeighborIndex",
            "DistanceMetric",
        ]:
            cls = getattr(baux, name)
            assert inspect.isclass(cls), f"baux.{name} should be a class"

    def test_evictedsymbols(self):
        import baux

        for name in REMOVED_LEGACY_SYMBOLS:
            assert name not in baux.__all__, f"{name} should not be in baux.__all__"
            assert name not in baux._LAZY_ATTRS, (
                f"{name} should not be in baux._LAZY_ATTRS"
            )
            with pytest.raises(AttributeError):
                getattr(baux, name)

    def test_expnamespace(self):
        import baux
        import baux.experimental as experimental_mod
        from baux import experimental

        assert inspect.ismodule(experimental)
        assert experimental is experimental_mod
        assert experimental.experimental is experimental

        # Verify experimental __all__ matches _LAZY_ATTRS exactly
        assert set(experimental.__all__) == set(experimental_mod._LAZY_ATTRS.keys()) | {
            "experimental"
        }

        # Evicted symbols from experimental
        evicted_exp = [
            "enn_optimizer",
            "ENN",
            "create_optimizer",
            "create_zero",
            "create_lhd",
            "turbo_enn",
        ]
        exp_all = getattr(experimental, "__all__", [])
        for name in evicted_exp:
            assert name not in exp_all, (
                f"{name} should not be in baux.experimental.__all__"
            )
            assert (
                not hasattr(experimental, name)
                or getattr(experimental, name) is None
                or getattr(experimental, name) is getattr(baux, name, object())
            ), f"baux.experimental.{name} should not be exposed"

        # Valid symbols in experimental
        assert inspect.isclass(experimental.SharingPolicy)
        assert inspect.isclass(experimental.RegionalSearchLoop)
        assert callable(experimental.make_region)
        assert callable(experimental.allocate_batches)
        assert callable(experimental.select_candidates)
        assert inspect.ismodule(experimental.multi_region)
        assert inspect.isclass(experimental.BayesOptimizer)
        assert inspect.isclass(experimental.MultiTrustRegion)

        # Check optree-dependent program symbols if optree installed
        import importlib.util

        if importlib.util.find_spec("optree") is not None:
            assert inspect.isclass(experimental.Fbt)
            assert callable(experimental.compile)
            assert inspect.isclass(experimental.TensorSpec)
            assert inspect.isclass(experimental.Executable)
            assert inspect.isclass(experimental.Invocation)
            assert inspect.isclass(experimental.ResidentModel)
            assert inspect.isclass(experimental.ModelOutput)
            assert inspect.isclass(experimental.ModelInvocation)

        if experimental.Portfolio is not None:
            assert inspect.isclass(experimental.Portfolio)
        if experimental.Region is not None:
            assert inspect.isclass(experimental.Region)

    def test_multiregionerr(self):
        from baux.experimental.multi_region import RegionalSearchLoop

        class BrokenState:
            calls = 0

            def tell(self, x, y, y_var=None):
                self.calls += 1
                raise TypeError("failure inside tell")

        state = BrokenState()
        loop = RegionalSearchLoop(state)
        with pytest.raises(TypeError, match="failure inside tell"):
            loop.tell(np.ones((1, 2)), np.ones((1, 1)), np.ones((1, 1)))
        assert state.calls == 1

    def test_telemetrycfg(self):
        import baux

        telem = baux.Telemetry(
            dt_fit=0.1, dt_gen=0.2, dt_sel=0.3, dt_tell=0.4, num_candidates=100
        )
        assert isinstance(telem, baux.Telemetry)
        assert telem.dt_fit == 0.1
        assert telem.dt_gen == 0.2
        assert telem.dt_sel == 0.3
        assert telem.dt_tell == 0.4
        assert telem.num_candidates == 100

        cfg = baux.OptimizerConfig()
        assert isinstance(cfg, baux.OptimizerConfig)

    def test_optlifecycle(self):
        import baux

        domain = np.array([[-2.0, 2.0], [0.0, 10.0]])
        cfg = baux.OptimizerConfig()
        s0 = baux.init(domain=domain, config=cfg, seed=42)
        assert isinstance(s0, baux.OptimizerState)
        assert s0.dim == 2
        assert s0.num_observations == 0

        # ask
        s1, batch = baux.ask(s0, num_arms=2)
        assert isinstance(s1, baux.OptimizerState)
        assert isinstance(batch, baux.AskBatch)
        assert batch.points.shape == (2, 2)
        assert len(batch.ids) == 2
        assert isinstance(batch.ids[0], baux.AskId)
        # Verify immutability of s0
        assert s0.num_observations == 0

        # tell
        y = np.array([[1.0], [2.0]])
        s2, fitted_mu = baux.tell(s1, batch.points, y)
        assert fitted_mu is not None
        assert isinstance(s2, baux.OptimizerState)
        assert s2.num_observations == 2
        assert s1.num_observations == 0
        assert s0.num_observations == 0

    def test_ennlifecycle(self):
        import baux

        train_x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        train_y = np.array([[0.0], [2.0], [1.0]])

        # enn_init
        es0 = baux.enn_init(train_x, train_y)
        assert isinstance(es0, baux.ENNState)
        assert len(es0) == 3

        # append
        new_x = np.array([[0.2, 0.2]])
        new_y = np.array([[0.4]])
        es1 = baux.append(es0, new_x, new_y)
        assert isinstance(es1, baux.ENNState)
        assert len(es1) == 4
        assert len(es0) == 3  # immutability

        # fit
        params = baux.fit(es1, k=2, num_candidates=10, num_samples=5, seed=123)
        assert isinstance(params, baux.ENNParams)
        assert params.k_neighbors > 0

        # posterior
        query_x = np.array([[0.3, 0.3]])
        normal = baux.posterior(es1, query_x, params)
        assert normal.mu.shape == (1, 1)
        assert normal.se.shape == (1, 1)

        # sample
        draws, idx = baux.sample(es1, query_x, params, num_draws=3, seeds=[1, 2, 3])
        assert idx is not None
        assert draws.shape == (1, 1, 3)
