"""Pure functional state and transitions (`init`, `append`, `posterior`, `sample`, `fit`) for ENN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from ..state import _readonly, _resolveseed
from .enn_class import ENN
from .enn_fit import enn_fit as _ennfit
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .posterior_flags import PosteriorFlags


@dataclass(frozen=True)
class ENNState:
    """Immutable value token representing an Epistemic Nearest Neighbors dataset & index."""

    x: np.ndarray
    y: np.ndarray
    y_var: np.ndarray | None
    scale_x: bool
    options: tuple[tuple[str, Any], ...]
    _model: ENN

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_outputs(self) -> int:
        return int(self.y.shape[1])


def enn_init(
    train_x: Any,
    train_y: Any,
    train_yvar: Any = None,
    *,
    scale_x: bool = False,
    **options: Any,
) -> ENNState:
    """Construct an immutable `ENNState` without mutating caller arrays."""
    x = _readonly(train_x)
    y = _readonly(train_y)
    if y.ndim == 1:
        y = _readonly(y.reshape(-1, 1))
    yvar = None if train_yvar is None else _readonly(train_yvar)
    if yvar is not None and yvar.ndim == 1:
        yvar = _readonly(yvar.reshape(-1, 1))
    model = ENN(x, y, yvar, scale_x=scale_x, **options)
    return ENNState(
        x=x,
        y=y,
        y_var=yvar,
        scale_x=scale_x,
        options=tuple(sorted(options.items())),
        _model=model,
    )


def enn_append(
    state: ENNState,
    x: Any,
    y: Any,
    y_var: Any = None,
) -> ENNState:
    """Return a new `ENNState` with appended observations, leaving `state` untouched."""
    x_new = _readonly(x)
    y_new = _readonly(y)
    if y_new.ndim == 1:
        y_new = _readonly(y_new.reshape(-1, 1))
    yvar_new = None if y_var is None else _readonly(y_var)
    if yvar_new is not None and yvar_new.ndim == 1:
        yvar_new = _readonly(yvar_new.reshape(-1, 1))
    if (state.y_var is None) != (yvar_new is None):
        raise ValueError("y_var presence must match existing ENNState")
    cat_x = _readonly(np.vstack([state.x, x_new]))
    cat_y = _readonly(np.vstack([state.y, y_new]))
    cat_yvar = (
        None
        if state.y_var is None or yvar_new is None
        else _readonly(np.vstack([state.y_var, yvar_new]))
    )
    opts = dict(state.options)
    model = ENN(cat_x, cat_y, cat_yvar, scale_x=state.scale_x, **opts)
    return ENNState(
        x=cat_x,
        y=cat_y,
        y_var=cat_yvar,
        scale_x=state.scale_x,
        options=state.options,
        _model=model,
    )


def enn_posterior(
    state: ENNState,
    x: Any,
    params: ENNParams,
    *,
    flags: PosteriorFlags | None = None,
) -> ENNNormal:
    """Pure posterior query on an immutable `ENNState`."""
    return state._model.posterior(x, params, flags=flags)


def enn_sample(
    state: ENNState,
    x: Any,
    params: ENNParams,
    *,
    seeds: np.ndarray | list[int] | tuple[int, ...],
    flags: PosteriorFlags | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure posterior function draw on an immutable `ENNState`."""
    return state._model.posterior_draw(
        x, params, function_seeds=list(seeds), flags=flags
    )


def enn_fit(
    state: ENNState,
    *,
    k: int = 10,
    num_candidates: int = 100,
    num_samples: int = 10,
    seed: int | Generator | None = None,
) -> ENNParams:
    """Pure hyperparameter fit returning `ENNParams` for an immutable `ENNState`."""
    rng = (
        seed
        if isinstance(seed, Generator)
        else np.random.default_rng(_resolveseed(seed))
    )
    return _ennfit(
        state._model,
        k=k,
        num_candidates=num_candidates,
        num_samples=num_samples,
        rng=rng,
    )


__all__ = [
    "ENNState",
    "enn_append",
    "enn_fit",
    "enn_init",
    "enn_posterior",
    "enn_sample",
]
