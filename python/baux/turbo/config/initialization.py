"""Initialization strategy configuration for TuRBO optimizers."""

from __future__ import annotations

from dataclasses import dataclass


class InitStrategy:
    """Base class for optimizer initialization strategies."""


@dataclass(frozen=True)
class HybridInit(InitStrategy):
    """Hybrid initialization combining initial design with trust-region search."""


@dataclass(frozen=True)
class LHDOnlyInit(InitStrategy):
    """Latin hypercube design-only initialization without surrogate search."""


@dataclass(frozen=True)
class InitConfig:
    """Configuration for initial design sampling."""

    init_strategy: InitStrategy | None = None
    num_init: int | None = None

    def __post_init__(self) -> None:
        if self.init_strategy is not None and not isinstance(
            self.init_strategy, InitStrategy
        ):
            raise ValueError(
                f"init_strategy must be an InitStrategy, got {self.init_strategy!r}"
            )
        if self.num_init is not None and self.num_init <= 0:
            raise ValueError(f"num_init must be > 0, got {self.num_init}")

    def get_strategy(self) -> InitStrategy:
        """Return the configured initialization strategy or default to HybridInit."""
        if self.init_strategy is not None:
            return self.init_strategy
        return HybridInit()
