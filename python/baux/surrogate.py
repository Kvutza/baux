"""Surrogate model configurations, neighbor indices, and metric learning policies."""

from __future__ import annotations

from baux.enn.posterior_flags import DiagonalMetric, MetricLearning
from baux.turbo.config.surrogate import (
    INDEX_NAMES,
    DistanceMetric,
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    NeighborIndex,
    NoSurrogate,
    Storage,
    Surrogate,
    SurrogateFit,
)

__all__ = [
    "INDEX_NAMES",
    "DiagonalMetric",
    "DistanceMetric",
    "ENNSurrogate",
    "ExternalSurrogate",
    "GaussianProcess",
    "MetricLearning",
    "NeighborIndex",
    "NoSurrogate",
    "Storage",
    "Surrogate",
    "SurrogateFit",
]
