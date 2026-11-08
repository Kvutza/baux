"""Data structures and sharing policy enums for multi-trust-region search."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NamedTuple


class SharingPolicy(StrEnum):
    """Observation-sharing policy across concurrent trust regions."""

    BOX_CONTAIN = "box_contain"
    ELLIPSOID_BALL = "ellipsoid_ball"
    NEAREST_CENTER = "nearest_center"
    ORIGIN_ONLY = "origin_only"


class RegionBatch(NamedTuple):
    """Contiguous candidate slice `(start, length)` allocated to `region`."""

    region: int
    start: int
    length: int


class RegionCandidate(NamedTuple):
    """Scored candidate proposal associated with a region and RNG seed."""

    index: int
    region: int
    seed: int
    score: float


class CandidateProposal(NamedTuple):
    """Unscored candidate payload proposed by `region` under `seed`."""

    region: int
    seed: int
    payload: Any
