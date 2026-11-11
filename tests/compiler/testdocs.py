"""Execute the public compiler examples against the packaged native runtime."""

import doctest
import importlib

import pytest


@pytest.mark.parametrize("name", ["fbt", "program"])
def test_examples(name):
    module = importlib.import_module(f"baux.experimental.{name}")
    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0
