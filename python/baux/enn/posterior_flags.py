"""Metric learning diagnostics and posterior evaluation flags for ENN."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MetricLearning(Enum):
    """Mode for automatic diagonal metric learning in ENN."""

    NONE = "none"
    AUTO = "auto"


@dataclass(frozen=True)
class DiagonalMetric:
    """Learned diagonal input metric weights and rebuild diagnostics."""

    num_seen: int
    heldout_gain: float | None
    uses_learned_metric: bool
    weights: tuple[float, ...]
    built: tuple[float, ...]
    num_refits: int = 0
    num_rescales: int = 0
    num_rebuilds: int = 0

    @property
    def scale(self) -> tuple[float, ...]:
        """Return per-dimension linear coordinate scales (square root of weights)."""
        return tuple(float(w) ** 0.5 for w in self.weights)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> DiagonalMetric | None:
        """Construct a DiagonalMetric instance from a Rust dictionary payload."""
        if raw is None:
            return None
        gain = raw.get("heldout_gain")
        return cls(
            num_seen=int(raw["num_seen"]),
            heldout_gain=None if gain is None else float(gain),
            uses_learned_metric=bool(raw["uses_learned_metric"]),
            weights=tuple(float(w) for w in raw.get("weights", ())),
            built=tuple(float(w) for w in raw.get("built", ())),
            num_refits=int(raw.get("num_refits", 0)),
            num_rescales=int(raw.get("num_rescales", 0)),
            num_rebuilds=int(raw.get("num_rebuilds", 0)),
        )


def metric_options(options: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize metric-learning keyword options for the Rust ENN backend."""
    ml = options.get("metric_learning")
    tied = options.get("tied_dims")
    unscaled = options.get("unscaled_dims")
    seed = options.get("metric_seed")
    if ml is None and tied is None and unscaled is None and seed is None:
        return None
    out: dict[str, Any] = {}
    if ml is not None:
        out["metric_learning"] = str(getattr(ml, "value", ml)).lower()
    if tied is not None:
        out["tied_dims"] = [[int(col) for col in group] for group in tied]
    if unscaled is not None:
        out["unscaled_dims"] = [int(col) for col in unscaled]
    if seed is not None:
        out["metric_seed"] = int(seed)
    return out


@dataclass(frozen=True)
class PosteriorFlags:
    """Flags controlling neighbor exclusion, observation noise, and tie handling."""

    exclude_nearest: bool = False
    observation_noise: bool = False
    tie_neighbors: bool = True
