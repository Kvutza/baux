"""Acquisition optimizer configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AcquisitionKind(Enum):
    """Supported acquisition function kinds."""

    THOMPSON = "thompson"
    PARETO = "pareto"
    UCB = "ucb"


@dataclass(frozen=True)
class RAASPOptimizerConfig:
    """Configuration for RAASP acquisition optimization."""


@dataclass(frozen=True)
class NDSOptimizerConfig:
    """Configuration for non-dominated sorting acquisition optimization."""


AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig
