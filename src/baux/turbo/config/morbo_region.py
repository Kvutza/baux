from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Rescalarize(Enum):
    ON_RESTART = "on_restart"
    ON_PROPOSE = "on_propose"


@dataclass(frozen=True)
class TrustRegionBounds:
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6

    def __post_init__(self) -> None:
        if self.length_init <= 0:
            raise ValueError(f"length_init must be > 0, got {self.length_init}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be > 0, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be > 0, got {self.length_max}")
        if self.length_min >= self.length_max:
            raise ValueError(
                f"length_min must be < length_max, got {self.length_min} >= {self.length_max}"
            )
        if self.length_init > self.length_max:
            raise ValueError(
                f"length_init must be <= length_max, got {self.length_init} > {self.length_max}"
            )
        if self.length_min > self.length_init:
            raise ValueError(
                f"length_min must be <= length_init, got {self.length_min} > {self.length_init}"
            )


@dataclass(frozen=True)
class MultiObjectiveConfig:
    num_metrics: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.num_metrics < 2:
            raise ValueError(
                f"num_metrics must be >= 2 for MORBO, got {self.num_metrics}"
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


@dataclass(frozen=True)
class RescalePolicy:
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE
