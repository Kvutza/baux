"""Candidate generation and sampling configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, Protocol

RestartCenterSelection = Literal["first", "maximin"]


@dataclass(frozen=True)
class RestartCenterPolicy:
    """Policy for selecting trust-region restart centers."""

    candidates: int = 1
    selection: RestartCenterSelection = "first"

    def __post_init__(self) -> None:
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int):
            raise TypeError("restart center candidates must be an integer")
        if self.candidates <= 0:
            raise ValueError("restart center candidates must be positive")
        if self.selection not in ("first", "maximin"):
            raise ValueError("unknown restart center selection")


class CandidateSampler(Enum):
    """Sampling method for generating trust-region candidates."""

    SOBOL = "sobol"
    UNIFORM = "uniform"
    RAASP = "raasp"


class RAASPDriver(Enum):
    """Execution driver for RAASP candidate generation."""

    ORIG = auto()
    FAST = auto()


if TYPE_CHECKING:

    class CandidateCount(Protocol):
        """Callable protocol returning candidate count for a given dimension and arm count."""

        def __call__(self, *, num_dim: int, num_arms: int) -> int:
            """Resolve candidate count from problem dimension and batch size."""
            ...
else:
    CandidateCount = Any


def default_candidates(*, num_dim: int, num_arms: int) -> int:
    """Compute default candidate count from problem dimensionality."""
    return min(5000, 100 * int(num_dim))


def fixed_candidates(count: int) -> CandidateCount:
    """Return a candidate count resolver that always returns a fixed count."""
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    def resolve(*, num_dim: int, num_arms: int) -> int:
        return count

    return resolve


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for trust-region candidate sampling."""

    sampler: CandidateSampler = CandidateSampler.SOBOL
    num_candidates: int | None = None
    num_candidates_per_arm: int | None = None
    num_pert: int | None = None
    raasp_driver: RAASPDriver = RAASPDriver.ORIG
    tied_dims: tuple[tuple[int, ...], ...] | None = None

    def resolve_candidates(self, *, num_dim: int, num_arms: int) -> int:
        """Compute total number of candidates for the given dimension and arm count."""
        base = (
            self.num_candidates
            if self.num_candidates is not None
            else default_candidates(num_dim=num_dim, num_arms=num_arms)
        )
        if self.num_candidates_per_arm is not None:
            base = max(base, self.num_candidates_per_arm * int(num_arms))
        return int(base)

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, CandidateSampler):
            raise TypeError(f"sampler must be a CandidateSampler, got {self.sampler!r}")
        for name, value in (
            ("num_candidates", self.num_candidates),
            ("num_candidates_per_arm", self.num_candidates_per_arm),
            ("num_pert", self.num_pert),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
