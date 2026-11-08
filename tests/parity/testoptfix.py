from __future__ import annotations

import tomllib

import pytest
from baux.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    assert_invariants,
    load_fixture,
)

try:
    load_fixture(EXPECTED_OPTIMIZER_FIXTURE_NAMES[0])
except tomllib.TOMLDecodeError:
    pytest.skip("optimizer fixtures require git lfs pull", allow_module_level=True)


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_001(name: str):
    data = load_fixture(name)
    assert_invariants(data)
