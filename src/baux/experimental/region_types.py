from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple


class SharingPolicy(str, Enum):
    BOX_CONTAIN = "box_contain"
    ELLIPSOID_BALL = "ellipsoid_ball"
    NEAREST_CENTER = "nearest_center"
    ORIGIN_ONLY = "origin_only"


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
