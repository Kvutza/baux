"""Pure functional state and transitions (`init`, `append`, `posterior`, `sample`, `fit`, `with_geometry`) for ENN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .._rust import ENNParams as _RustENNParams
from .._rust import fit_model as _fitmodel
from ..pytree import Affine, Log, Periodic, Simplex, Space, Stiefel
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
        """Return input feature dimension."""
        return int(self.x.shape[1])

    @property
    def num_outputs(self) -> int:
        """Return number of output metrics."""
        return int(self.y.shape[1])

    @property
    def key(self) -> Any:
        """Return content-addressed deterministic PRNG key for this state."""
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
    out._metric = (
        getattr(base, "_metric", ("euclidean", 4)) if metric is None else metric
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
    from dataclasses import replace as _replace

    from ..state import OptimizerState
    from ..transforms import Transform
    from ..turbo.config.optimizer import OptimizerConfig
    from ..turbo.config.surrogate import ENNSurrogate

    if isinstance(model, OptimizerConfig):
        if not isinstance(model.surrogate, ENNSurrogate):
            raise ValueError("with_geometry on OptimizerConfig requires ENNSurrogate")
        return _replace(
            model, surrogate=_replace(model.surrogate, geometry=m, local_k=k)
        )
    if isinstance(model, Transform):
        return model | Transform(
            f"with_geometry({m!r}, local_k={k})",
            lambda cfg: with_geometry(cfg, m, k),
        )
    if isinstance(model, OptimizerState):
        from ..functional import init as _opt_init

        cfg = with_geometry(model.config, m, k)
        return _opt_init(
            model.spec if model.spec is not None else model.bounds,
            cfg,
            seed=model.seed,
        )
    if hasattr(model, "_rust") and hasattr(model._rust, "with_geometry"):
        model._rust = model._rust.with_geometry(m, k)
    elif hasattr(model, "with_geometry") and callable(model.with_geometry):
        return model.with_geometry(m, k)
    model._geometry = m, k
    return model


def _stiefelarr(stiefel: Any) -> np.ndarray | None:
    if stiefel is None:
        return None
    return (
        stiefel.loadings
        if isinstance(stiefel, Stiefel)
        else np.asarray(stiefel, dtype=float)
    )


def _affinetup(affine: Any) -> tuple[list[float], list[float], list[float]] | None:
    if affine is None:
        return None
    a, b, c = (affine.a, affine.b, affine.c) if isinstance(affine, Affine) else affine
    return (
        np.asarray(a, dtype=float).ravel().tolist(),
        np.asarray(b, dtype=float).ravel().tolist(),
        np.asarray(c, dtype=float).ravel().tolist(),
    )


def _fitcodomain(
    state: ENNState,
    stiefel: Any,
    affine: Any,
    stiefel_rank: int | None,
    calibrate_alpha: float | None,
) -> tuple[Any, Any]:
    if stiefel is None and stiefel_rank is not None:
        stiefel = Stiefel.from_cov(state.y, rank=int(stiefel_rank))
    if affine is None and calibrate_alpha is not None:
        k_fit = min(10, max(1, len(state) - 1))
        p = enn_fit(state, k=k_fit, num_candidates=20, num_samples=5)
        loo = enn_posterior(
            state, state.x, p, flags=PosteriorFlags(exclude_nearest=True)
        )
        aff = Affine.fit_wls(loo.mu, state.y, loo.se)
        if 0.0 < float(calibrate_alpha) < 1.0:
            c_q = aff.conformal_scale(
                loo.mu, state.y, loo.se, alpha=float(calibrate_alpha)
            )
            aff = Affine(aff.a, aff.b, c_q)
        affine = aff
    return stiefel, affine


def _codomain(
    target: Any,
    *,
    stiefel: Any = None,
    affine: Any = None,
    stiefel_rank: int | None = None,
    calibrate_alpha: float | None = None,
) -> Any:
    if isinstance(target, ENNState):
        stiefel, affine = _fitcodomain(
            target, stiefel, affine, stiefel_rank, calibrate_alpha
        )
        st_arr, aff_tup = _stiefelarr(stiefel), _affinetup(affine)
        rust_obj = target._model._rust_model.append_cow(stiefel=st_arr, affine=aff_tup)
        opts = dict(target.options)
        if st_arr is not None:
            opts["stiefel"] = st_arr.tobytes()
        if aff_tup is not None:
            opts["affine"] = aff_tup
        return ENNState(
            x=target.x,
            y=target.y,
            y_var=target.y_var,
            scale_x=target.scale_x,
            options=tuple(sorted(opts.items())),
            _model=_wrapenn(target._model, rust_obj),
        )
    if isinstance(target, ENN):
        st_arr, aff_tup = _stiefelarr(stiefel), _affinetup(affine)
        return _wrapenn(
            target, target._rust_model.append_cow(stiefel=st_arr, affine=aff_tup)
        )
    from dataclasses import replace as _replace

    from ..state import OptimizerState
    from ..transforms import Transform
    from ..turbo.config.optimizer import OptimizerConfig
    from ..turbo.config.surrogate import ENNSurrogate

    if isinstance(target, OptimizerConfig):
        if not isinstance(target.surrogate, ENNSurrogate):
            raise ValueError(
                "codomain transform on OptimizerConfig requires ENNSurrogate"
            )
        kw = {
            k: v
            for k, v in (
                ("stiefel", stiefel),
                ("affine", affine),
                ("stiefel_rank", stiefel_rank),
                ("calibrate_alpha", calibrate_alpha),
            )
            if v is not None
        }
        return _replace(target, surrogate=_replace(target.surrogate, **kw))
    if isinstance(target, Transform):
        label = (
            "stein()"
            if (stiefel is not None or stiefel_rank is not None)
            else "calibrate()"
        )
        return target | Transform(
            label,
            lambda cfg: _codomain(
                cfg,
                stiefel=stiefel,
                affine=affine,
                stiefel_rank=stiefel_rank,
                calibrate_alpha=calibrate_alpha,
            ),
        )
    if isinstance(target, OptimizerState):
        from ..functional import init as _opt_init

        cfg = _codomain(
            target.config,
            stiefel=stiefel,
            affine=affine,
            stiefel_rank=stiefel_rank,
            calibrate_alpha=calibrate_alpha,
        )
        return _opt_init(
            target.spec if target.spec is not None else target.bounds,
            cfg,
            seed=target.seed,
        )
    raise TypeError(f"Unsupported target for codomain transform: {type(target)!r}")


def stein(target: Any = None, frame: Any = None, *, rank: int | None = None) -> Any:
    """Applies Stiefel subspace James-Stein contraction to a state, config, or Transform.

    Args:
        target: Target state/config/Transform, or a Stiefel frame / integer rank
            when constructing a pipeline Transform.
        frame: Optional Stiefel frame or integer rank when target is a state or config.
        rank: Optional principal Stiefel subspace rank for online estimation.

    Returns:
        Updated state/config or composable Transform.
    """
    from ..transforms import Transform

    if target is None:
        r = 1 if rank is None else int(rank)
        return Transform(f"stein(rank={r})", lambda cfg: _codomain(cfg, stiefel_rank=r))
    if isinstance(target, (int, np.integer)):
        r = int(target)
        return Transform(f"stein(rank={r})", lambda cfg: _codomain(cfg, stiefel_rank=r))
    if frame is None and rank is None:
        return Transform("stein()", lambda cfg: _codomain(cfg, stiefel=target))
    if rank is not None:
        return _codomain(target, stiefel_rank=int(rank))
    if isinstance(frame, (int, np.integer)):
        return _codomain(target, stiefel_rank=int(frame))
    return _codomain(target, stiefel=frame)


def calibrate(
    target: Any = None, affine: Any = None, *, alpha: float | None = None
) -> Any:
    """Applies an Affine posterior location-scale map to a state, config, or Transform.

    Args:
        target: Target state/config/Transform, or an Affine map / conformal alpha
            when constructing a pipeline Transform.
        affine: Optional Affine map or conformal alpha when target is a state or config.
        alpha: Optional conformal miscoverage rate in (0, 1) for online calibration.

    Returns:
        Updated state/config or composable Transform.
    """
    from ..transforms import Transform

    if target is None:
        a = 0.0 if alpha is None else float(alpha)
        return Transform("calibrate()", lambda cfg: _codomain(cfg, calibrate_alpha=a))
    if isinstance(target, (int, float, np.floating)):
        a = float(target)
        return Transform("calibrate()", lambda cfg: _codomain(cfg, calibrate_alpha=a))
    if affine is None and alpha is None and isinstance(target, (Affine, tuple, list)):
        return Transform("calibrate()", lambda cfg: _codomain(cfg, affine=target))
    if alpha is not None:
        return _codomain(target, calibrate_alpha=float(alpha))
    if isinstance(affine, (int, float, np.floating)):
        return _codomain(target, calibrate_alpha=float(affine))
    if affine is None:
        return _codomain(target, calibrate_alpha=0.0)
    return _codomain(target, affine=affine)


def enn_init(
    train_x: Any,
    train_y: Any,
    train_yvar: Any = None,
    *,
    scale_x: bool = False,
    stiefel: Stiefel | np.ndarray | None = None,
    affine: Affine | tuple[Any, Any, Any] | None = None,
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
    st_arr, aff_tup = _stiefelarr(stiefel), _affinetup(affine)
    if st_arr is not None or aff_tup is not None:
        model = _wrapenn(
            model, model._rust_model.append_cow(stiefel=st_arr, affine=aff_tup)
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
    "Affine",
    "ENNState",
    "Log",
    "Periodic",
    "Simplex",
    "Space",
    "Stiefel",
    "append",
    "calibrate",
    "enn_append",
    "enn_fit",
    "enn_init",
    "enn_posterior",
    "enn_sample",
    "fit",
    "posterior",
    "sample",
    "stein",
    "with_geometry",
]
