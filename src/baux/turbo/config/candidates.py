from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol


class CandidateSampler(Enum):
    SOBOL = "sobol"
    UNIFORM = "uniform"
    RAASP = "raasp"


class RAASPDriver(Enum):
    ORIG = auto()
    FAST = auto()


if TYPE_CHECKING:

    class CandidateCount(Protocol):
        def __call__(self, *, num_dim: int, num_arms: int) -> int: ...
else:
    CandidateCount = Any


def default_candidates(*, num_dim: int, num_arms: int) -> int:
    return min(5000, 100 * int(num_dim))


def fixed_candidates(count: int) -> CandidateCount:
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    def resolve(*, num_dim: int, num_arms: int) -> int:
        return count

    return resolve


@dataclass(frozen=True)
class CandidateConfig:
    sampler: CandidateSampler = CandidateSampler.SOBOL
    num_candidates: int | None = None
    num_candidates_per_arm: int | None = None
    num_pert: int | None = None
    raasp_driver: RAASPDriver = RAASPDriver.ORIG

    def resolve_candidates(self, *, num_dim: int, num_arms: int) -> int:
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
