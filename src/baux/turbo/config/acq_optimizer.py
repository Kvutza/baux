from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AcquisitionKind(Enum):
    THOMPSON = "thompson"
    PARETO = "pareto"
    UCB = "ucb"


@dataclass(frozen=True)
class RAASPOptimizerConfig:
    pass


@dataclass(frozen=True)
class NDSOptimizerConfig:
    pass


AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig
