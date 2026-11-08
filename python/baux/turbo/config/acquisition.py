"""Acquisition function configuration types."""

from __future__ import annotations

from dataclasses import dataclass

from .acq_optimizer import (
    AcqOptimizerConfig,
    AcquisitionKind,
    NDSOptimizerConfig,
    RAASPOptimizerConfig,
)


@dataclass(frozen=True)
class DrawAcquisitionConfig:
    """Configuration for posterior draw (Thompson sampling) acquisition."""


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    """Configuration for Pareto front acquisition."""


@dataclass(frozen=True)
class RandomAcquisitionConfig:
    """Configuration for uniform random candidate acquisition."""


@dataclass(frozen=True)
class UCBAcquisitionConfig:
    """Configuration for Upper Confidence Bound acquisition."""

    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")


AcquisitionConfig = (
    UCBAcquisitionConfig
    | DrawAcquisitionConfig
    | ParetoAcquisitionConfig
    | RandomAcquisitionConfig
)

__all__ = [
    "AcqOptimizerConfig",
    "AcquisitionConfig",
    "AcquisitionKind",
    "DrawAcquisitionConfig",
    "NDSOptimizerConfig",
    "ParetoAcquisitionConfig",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "UCBAcquisitionConfig",
]
