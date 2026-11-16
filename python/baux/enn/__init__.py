"""Functional Epistemic Nearest Neighbors (ENN) namespace."""

from __future__ import annotations

import sys
import types
from typing import Any

from ..pytree import Log, Periodic, Simplex, Space
from ..transforms import Transform
from ..transforms import enn as _enntransform
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .enn_state import (
    Affine,
    ENNState,
    Stiefel,
    append,
    calibrate,
    enn_init,
    ensure_sync,
    fit,
    persist_index,
    posterior,
    sample,
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
    "enn_init",
    "ensure_sync",
    "fit",
    "persist_index",
    "posterior",
    "sample",
    "stein",
    "with_geometry",
]
