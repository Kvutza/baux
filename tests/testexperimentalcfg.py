import pytest

from baux import experimental


@pytest.mark.parametrize("settings", [{}, {"sampler": "gaussian", "length_init": 0.1}])
def test_settings(monkeypatch, settings):
    calls = []

    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return "search"

    monkeypatch.setattr(experimental, "__getattr__", lambda _: constructor)
    assert (
        experimental.turbo_enn("base", -1.0, [], 4, reference_seed=23, **settings)
        == "search"
    )
    expected = {
        "max_pending": 1,
        "base_variance": 0.0,
        "length_init": 0.8,
        "length_min": 0.0078125,
        "length_max": 1.6,
        "failure_tolerance": None,
        "sampler": "independent",
        "reference_seed": 23,
    }
    expected.update(settings)
    assert calls == [(("base", -1.0, [], 4), expected)]


def test_unknown():
    with pytest.raises(TypeError, match="unexpected keyword"):
        experimental.turbo_enn("base", -1.0, [], 4, unknown_scale=0.5)
