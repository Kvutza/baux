import pytest
from baux import experimental
from baux.turbo.config import turbo_enn


def test_expremoved():
    for name in (
        "ENN",
        "create_optimizer",
        "create_zero",
        "create_lhd",
        "enn_optimizer",
        "turbo_enn",
    ):
        assert not hasattr(experimental, name)
        assert name not in getattr(experimental, "__all__", [])
        assert name not in dir(experimental)
        with pytest.raises(AttributeError):
            getattr(experimental, name)


def test_turboenncfg():
    cfg = turbo_enn()
    assert cfg is not None
