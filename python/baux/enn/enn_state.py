"""Pure functional state and transitions (`init`, `append`, `posterior`, `sample`, `fit`, `with_geometry`) for ENN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .._rust import ENNParams as _RustENNParams
from .._rust import fit_model as _fitmodel
from ..pytree import Calibrator, Log, Periodic, Shrinkage, Simplex, Space
from ..state import _readonly, _resolveseed
from .enn_class import ENN
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .posterior_flags import PosteriorFlags

_METRICS = ("euclidean", "conformal", "hyperbolic", "wasserstein")


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

    @property
    def key(self) -> Any:
        from .._rust import Key as _Key

        buf = (
            np.ascontiguousarray(self.x, dtype=float).tobytes()
            + np.ascontiguousarray(self.y, dtype=float).tobytes()
            + (
                b""
                if self.y_var is None
                else np.ascontiguousarray(self.y_var, dtype=float).tobytes()
            )
            + repr(self.options).encode("utf-8")
        )
        return _Key.from_bytes(buf)


def _wrapenn(base: ENN, rust_model: Any, metric: tuple[str, int] | None = None) -> ENN:
    out = object.__new__(ENN)
    out._index_driver = base._index_driver
    out._rust_model = rust_model
    out.device = base.device
    out.index_kind = base.index_kind
    setattr(
        out,
        "_metric",
        getattr(base, "_metric", ("euclidean", 4)) if metric is None else metric,
    )
    return out


def with_geometry(model: Any, geometry: str, local_k: int = 4) -> Any:
    """Return a model or `ENNState` configured with a local-scale Riemannian dissimilarity geometry."""
    m = str(geometry).strip().lower()
    k = int(local_k)
    if m not in _METRICS or k < 1:
        raise ValueError(f"invalid geometry={geometry!r} or local_k={local_k}")
    if isinstance(model, ENNState):
        rust_obj = model._model._rust_model.append_cow(metric=m, local_k=k)
        opts = dict(model.options)
        opts["geometry"] = m
        opts["local_k"] = k
        return ENNState(
            x=model.x,
            y=model.y,
            y_var=model.y_var,
            scale_x=model.scale_x,
            options=tuple(sorted(opts.items())),
            _model=_wrapenn(model._model, rust_obj, (m, k)),
        )
    if isinstance(model, ENN):
        return _wrapenn(
            model, model._rust_model.append_cow(metric=m, local_k=k), (m, k)
        )
    if hasattr(model, "_rust") and hasattr(model._rust, "with_geometry"):
        model._rust = model._rust.with_geometry(m, k)
    elif hasattr(model, "with_geometry") and callable(model.with_geometry):
        return model.with_geometry(m, k)
    setattr(model, "_geometry", (m, k))
    return model


def enn_init(
    train_x: Any,
    train_y: Any,
    train_yvar: Any = None,
    *,
    scale_x: bool = False,
    shrinkage: Shrinkage | np.ndarray | None = None,
    calibrator: Calibrator | tuple[Any, Any, Any] | None = None,
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
    sh_arr = (
        None
        if shrinkage is None
        else (
            shrinkage.loadings
            if isinstance(shrinkage, Shrinkage)
            else np.asarray(shrinkage, dtype=float)
        )
    )
    cal_tup = None
    if calibrator is not None:
        a, b, c = (
            (calibrator.a, calibrator.b, calibrator.c)
            if isinstance(calibrator, Calibrator)
            else calibrator
        )
        cal_tup = (
            np.asarray(a, dtype=float).ravel().tolist(),
            np.asarray(b, dtype=float).ravel().tolist(),
            np.asarray(c, dtype=float).ravel().tolist(),
        )
    if sh_arr is not None or cal_tup is not None:
        model = _wrapenn(
            model, model._rust_model.append_cow(shrinkage=sh_arr, calibrator=cal_tup)
        )
    return ENNState(
        x=x,
        y=y,
        y_var=yvar,
        scale_x=scale_x,
        options=tuple(sorted(options.items())),
        _model=model,
    )


def enn_append(state: ENNState, x: Any, y: Any, y_var: Any = None) -> ENNState:
    """Return a new `ENNState` with appended observations via Rust COW, leaving `state` untouched."""
    x_new = _readonly(x)
    if x_new.ndim == 1:
        x_new = _readonly(x_new.reshape(1, -1))
    y_new = _readonly(y)
    if y_new.ndim == 1:
        y_new = _readonly(y_new.reshape(-1, 1))
    yvar_new = None if y_var is None else _readonly(y_var)
    if yvar_new is not None and yvar_new.ndim == 1:
        yvar_new = _readonly(yvar_new.reshape(-1, 1))
    if (state.y_var is None) != (yvar_new is None):
        raise ValueError("y_var presence must match existing ENNState")
    x_in, y_in, yvar_in = state._model._inputs(x_new, y_new, yvar_new)
    model = _wrapenn(
        state._model, state._model._rust_model.append_cow(x_in, y_in, yvar_in)
    )
    cat_x = _readonly(np.vstack([state.x, x_new]))
    cat_y = _readonly(np.vstack([state.y, y_new]))
    cat_yvar = (
        None
        if state.y_var is None or yvar_new is None
        else _readonly(np.vstack([state.y_var, yvar_new]))
    )
    return ENNState(
        x=cat_x,
        y=cat_y,
        y_var=cat_yvar,
        scale_x=state.scale_x,
        options=state.options,
        _model=model,
    )


def enn_posterior(
    state: ENNState, x: Any, params: ENNParams, *, flags: PosteriorFlags | None = None
) -> ENNNormal:
    """Pure posterior query on an immutable `ENNState`."""
    return state._model.posterior(x, params, flags=flags)


def enn_sample(
    state: ENNState,
    x: Any,
    params: ENNParams,
    *,
    key: Any = None,
    num_draws: int = 1,
    seeds: np.ndarray | list[int] | tuple[int, ...] | None = None,
    flags: PosteriorFlags | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure posterior function draw on an immutable `ENNState`."""
    if seeds is None:
        from .._rust import Key as _Key

        if key is None:
            qx = _Key.from_bytes(np.ascontiguousarray(x, dtype=float).tobytes())
            root = state.key.fold(qx.seed())
        else:
            root = key if isinstance(key, _Key) else _Key(key)
        fn_seeds = [
            int(sk.seed()) & ((1 << 63) - 1) for sk in root.split_n(int(num_draws))
        ]
    else:
        fn_seeds = list(seeds)
    return state._model.posterior_draw(x, params, function_seeds=fn_seeds, flags=flags)


def enn_fit(
    state: ENNState,
    k: int = 10,
    *,
    num_candidates: int = 100,
    num_samples: int = 10,
    infer_aleatoric: bool = True,
    key: Any = None,
    seed: int | Generator | None = None,
    warm: ENNParams | None = None,
) -> ENNParams:
    """Pure stateless hyperparameter fit returning `ENNParams` for an immutable `ENNState`."""
    default_key = (
        state.key.fold(int(k)).fold(int(num_samples))
        if (seed is None and key is None)
        else key
    )
    fit_seed = _resolveseed(seed, default_key)
    rust_warm = (
        None
        if warm is None
        else _RustENNParams(
            warm.k_neighbors, warm.epistemic_scale, warm.aleatoric_scale
        )
    )
    res = _fitmodel(
        state._model.rust_backend,
        int(k),
        int(num_candidates),
        int(num_samples),
        bool(infer_aleatoric),
        int(fit_seed),
        rust_warm,
    )
    return ENNParams(
        k_neighbors=res.k_neighbors,
        epistemic_scale=res.epistemic_scale,
        aleatoric_scale=res.aleatoric_scale,
    )


append = enn_append
fit = enn_fit
posterior = enn_posterior
sample = enn_sample


__all__ = [
    "Calibrator",
    "ENNState",
    "Log",
    "Periodic",
    "Shrinkage",
    "Simplex",
    "Space",
    "append",
    "enn_append",
    "enn_fit",
    "enn_init",
    "enn_posterior",
    "enn_sample",
    "fit",
    "posterior",
    "sample",
    "with_geometry",
]
