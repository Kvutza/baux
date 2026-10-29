from __future__ import annotations

import pytest

from baux.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    load_fixture,
    replay_check,
)
from baux.turbo.optimizer_fixtures.replay import _config

pytest.importorskip("baux._rust")


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_001(name: str):
    data = load_fixture(name)
    config = _config(name)
    replay_check(data, config)
