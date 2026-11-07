"""Functional & object-oriented Epistemic Nearest Neighbors (ENN) namespace."""

from __future__ import annotations

import sys
import types
from typing import Any

from ..transforms import Transform, enn as _enntransform
from .enn_class import ENN, Enn
from .enn_fitter import ENNFitter
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .enn_state import (
    ENNState,
    enn_append as append,
    enn_fit as fit,
    enn_init as init,
    enn_posterior as posterior,
    enn_sample as sample,
)
from .posterior_flags import DiagonalMetric, MetricLearning, PosteriorFlags


class _EnnModule(types.ModuleType):
    def __call__(self, **kwargs: Any) -> Transform:
        return _enntransform(**kwargs)


sys.modules[__name__].__class__ = _EnnModule
enn: Any = sys.modules[__name__]

__all__ = [
    "DiagonalMetric",
    "ENN",
    "ENNFitter",
    "ENNNormal",
    "ENNParams",
    "ENNState",
    "Enn",
    "MetricLearning",
    "PosteriorFlags",
    "append",
    "enn",
    "fit",
    "init",
    "posterior",
    "sample",
]
