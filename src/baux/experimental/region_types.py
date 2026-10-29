from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple


class SharingPolicy(str, Enum):
    SHARED = "shared"
    NEAREST_CENTER = "nearest_center"
    INDEPENDENT = "independent"


class RegionBatch(NamedTuple):
    region: int
    start: int
    length: int


class RegionCandidate(NamedTuple):
    index: int
    region: int
    seed: int
    score: float


class CandidateProposal(NamedTuple):
    region: int
    seed: int
    payload: Any
