from __future__ import annotations

import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("baux.baux_rust") is None,
    reason="baux_rust extension unavailable",
)


class TestPublicAPIExports:
    """Verify public API surface is exported correctly from baux package."""

    def test_001(self):
        """All _LAZY_ATTRS entries must be importable."""
        import baux

        for name in baux._LAZY_ATTRS:
            attr = getattr(baux, name)
            assert attr is not None, f"Attribute {name} not found"

    def test_002(self):
        """ENN is a class with expected signature."""
        from baux import ENN

        assert inspect.isclass(ENN)
        # Check constructor signature has required parameters
        sig = inspect.signature(ENN.__init__)
        params = list(sig.parameters.keys())
        assert "train_x" in params
        assert "train_y" in params
        assert "train_yvar" in params

    def test_003(self):
        from baux import ENNFitter

        assert ENNFitter is not None

    def test_004(self):
        """create_optimizer is a callable function."""
        from baux import create_optimizer

        assert callable(create_optimizer)

    def test_optimizer(self):
        import baux
        from baux import bo

        assert baux.Optimizer is bo.Optimizer
        assert baux.BayesOptimizer is baux.Optimizer
        assert baux.Optimizer.__name__ == "Optimizer"

    def test_configclasses(self):
        """All config classes are available and constructible."""
        from baux import (
            MorboTrustRegion,
            NoTrustRegion,
            OptimizerConfig,
            TurboTrustRegion,
        )

        # These are dataclasses or similar - should be constructible
        assert inspect.isclass(OptimizerConfig)
        assert inspect.isclass(TurboTrustRegion)
        assert inspect.isclass(MorboTrustRegion)
        assert inspect.isclass(NoTrustRegion)

    def test_005(self):
        """All config factory functions are callable."""
        from baux import (
            lhd_only,
            turbo_enn,
            turbo_one,
            turbo_zero,
        )

        assert callable(turbo_one)
        assert callable(turbo_zero)
        assert callable(turbo_enn)
        assert callable(lhd_only)

    def test_telemetryclass(self):
        """Telemetry class is available."""
        from baux import Telemetry

        assert inspect.isclass(Telemetry)

    def test_006(self):
        """Experimental namespace is importable and exposes low-level symbols."""
        import pytest

        pytest.importorskip("baux.baux_rust")
        import baux.experimental as experimental_mod
        from baux import experimental

        assert inspect.ismodule(experimental)
        assert experimental is experimental_mod
        assert experimental.experimental is experimental
        assert inspect.isclass(experimental.SharingPolicy)
        assert inspect.isclass(experimental.MultiTrustRegionLoop)
        assert callable(experimental.make_region)
        assert inspect.isclass(experimental.BayesOptimizer)
        assert inspect.isclass(experimental.MultiTrustRegion)
        assert inspect.isclass(experimental.ResidentBoSession)
        assert callable(experimental.enn_optimizer)
        assert callable(experimental.quantize_int4)
        assert callable(experimental.quantize_e2m1)
        for name in [
            "ParamBlock",
            "ParameterBuffer",
            "Proposals",
            "SearchState",
            "turbo_enn",
        ]:
            assert name in experimental.__all__
            assert hasattr(experimental, name)
        if experimental.SearchState is not None:
            assert inspect.isclass(experimental.ParamBlock)
            assert inspect.isclass(experimental.ParameterBuffer)
            assert experimental.ParameterBuffer is experimental.ParamBuffer
            assert inspect.isclass(experimental.Proposals)
            assert inspect.isclass(experimental.SearchState)
            assert callable(experimental.turbo_enn)

    def test_enumtypes(self):
        """Enum types are available."""
        from baux import AcquisitionKind, CandidateSampler, InitStrategy

        assert inspect.isclass(CandidateSampler)
        assert inspect.isclass(InitStrategy)
        assert inspect.isclass(AcquisitionKind)


class Test_001:
    """Verify public API does not change unexpectedly."""

    def test_007(self):
        """__all__ must match _LAZY_ATTRS keys."""
        import baux

        assert set(baux.__all__) == set(baux._LAZY_ATTRS.keys())

    def test_008(self):
        import baux

        assert "quantize_int4" not in baux.__all__
        assert "quantize_e2m1" not in baux.__all__

    def test_009(self):
        import baux

        for name in ("quantize_int4", "quantize_e2m1"):
            with pytest.raises(AttributeError):
                getattr(baux, name)
