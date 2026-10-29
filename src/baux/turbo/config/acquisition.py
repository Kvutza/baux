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
    pass


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    pass


@dataclass(frozen=True)
class RandomAcquisitionConfig:
    pass


@dataclass(frozen=True)
class UCBAcquisitionConfig:
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
