"""Functional & object-oriented Epistemic Nearest Neighbors (ENN) namespace."""

from __future__ import annotations

import sys
import types
from typing import Any

from ..pytree import Log, Periodic, Simplex, Space
from ..transforms import Transform, enn as _enntransform
from .enn_class import ENN
from .enn_fitter import ENNFitter
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .enn_state import (
    Affine,
    ENNState,
    Stiefel,
    calibrate,
    enn_append,
    enn_append as append,
    enn_fit,
    enn_fit as fit,
    enn_init,
    enn_init as init,
    enn_posterior,
    enn_posterior as posterior,
    enn_sample,
    enn_sample as sample,
    stein,
    with_geometry,
)
from .posterior_flags import DiagonalMetric, MetricLearning, PosteriorFlags


class _EnnModule(types.ModuleType):
    def __call__(self, **kwargs: Any) -> Transform:
        return _enntransform(**kwargs)


sys.modules[__name__].__class__ = _EnnModule
enn: Any = sys.modules[__name__]

__all__ = [
    "Affine",
    "DiagonalMetric",
    "ENN",
    "ENNFitter",
    "ENNNormal",
    "ENNParams",
    "ENNState",
    "Log",
    "MetricLearning",
    "Periodic",
    "PosteriorFlags",
    "Simplex",
    "Space",
    "Stiefel",
    "append",
    "calibrate",
    "enn",
    "enn_append",
    "enn_fit",
    "enn_init",
    "enn_posterior",
    "enn_sample",
    "fit",
    "init",
    "posterior",
    "sample",
    "stein",
    "with_geometry",
]
