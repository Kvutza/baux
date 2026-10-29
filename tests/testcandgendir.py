from __future__ import annotations

import pytest

from baux.turbo.config import CandidateConfig
from baux.turbo.config.candidates import (
    fixed_candidates,
    default_candidates,
)


def test_001():
    cfg = CandidateConfig()
    assert cfg.resolve_candidates(num_dim=1, num_arms=1) == 100
    assert cfg.resolve_candidates(num_dim=100, num_arms=1) == 5000


def test_002():
    assert default_candidates(num_dim=3, num_arms=2) == 300
    assert default_candidates(num_dim=100, num_arms=1) == 5000


def test_003():
    assert fixed_candidates(7)(num_dim=1, num_arms=1) == 7


def test_constnumcandidates():
    cfg = CandidateConfig(num_candidates=123)
    assert cfg.resolve_candidates(num_dim=3, num_arms=7) == 123
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        CandidateConfig(num_candidates=0)


def test_004():
    cfg = CandidateConfig(num_candidates_per_arm=50)
    assert cfg.resolve_candidates(num_dim=3, num_arms=4) == 300


def test_005():
    cfg = CandidateConfig(num_candidates=100, num_candidates_per_arm=50)
    assert cfg.resolve_candidates(num_dim=3, num_arms=4) == 200


def test_006():
    with pytest.raises(TypeError):
        CandidateConfig(num_candidates=fixed_candidates(5))  # type: ignore[arg-type]
