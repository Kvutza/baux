"""Pure functional state and transitions (`init`, `append`, `posterior`, `sample`, `fit`, `with_geometry`) for ENN."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from .._rust import ENN as _RustENN
from .._rust import ENNParams as _RustENNParams
from .._rust import fit_model as _fitmodel
from ..index import INDEX_NAMES, NeighborIndex
from ..pytree import Affine, Log, Periodic, Simplex, Space, Stiefel
from ..state import _readonly, _resolveseed
from .draw_internals import DrawInternals
from .enn_normal import ENNNormal
from .enn_params import ENNParams
from .neighbor_data import NeighborData
from .posterior_flags import DiagonalMetric, PosteriorFlags, metric_options

_METRICS = ("euclidean", "conformal", "hyperbolic", "wasserstein")

_DEVICE_DRIVERS = {
    "cpu": NeighborIndex.FLAT,
    "metal": NeighborIndex.METAL,
    "cuda": NeighborIndex.CUDA,
    "opencl": NeighborIndex.OPENCL,
    "webgpu": NeighborIndex.WEBGPU,
}
_INDEX_ALIASES = {
    "exact": NeighborIndex.FLAT,
    "flat": NeighborIndex.FLAT,
    "hnsw": NeighborIndex.USEARCH,
    "usearch": NeighborIndex.USEARCH,
    "tree": NeighborIndex.BPANN_DISK,
    "bpann": NeighborIndex.BPANN_DISK,
    "bpann_disk": NeighborIndex.BPANN_DISK,
    "auto": NeighborIndex.AUTO,
}
_INDEX_KINDS = {
    NeighborIndex.USEARCH: "hnsw",
    NeighborIndex.BPANN_DISK: "tree",
    NeighborIndex.AUTO: "auto",
}


def _rustseeds(function_seeds: Any) -> list[int]:
    if hasattr(function_seeds, "__iter__"):
        return [int(s) for s in np.asarray(function_seeds, dtype=np.int64).tolist()]
    return [int(function_seeds)]


def _finaldraw(
    x: np.ndarray, draws: np.ndarray, idx: list[Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(
            f"function draws must be 3D (num_samples, batch, metrics), got {draws_arr.shape}"
        )
    draws_arr = np.transpose(draws_arr, (1, 2, 0))
    idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0), dtype=int)
    return draws_arr, idx_arr


def _inputs(
    train_x: Any, train_y: Any, train_yvar: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    tx = np.asarray(train_x, dtype=float)
    ty = np.asarray(train_y, dtype=float)
    if ty.ndim == 1:
        ty = ty.reshape(-1, 1)
    if tx.ndim != 2 or ty.ndim != 2 or tx.shape[0] != ty.shape[0]:
        raise ValueError((tx.shape, ty.shape))
    tyvar = None
    if train_yvar is not None:
        tyvar = np.asarray(train_yvar, dtype=float)
        if tyvar.ndim == 1:
            tyvar = tyvar.reshape(-1, 1)
        if tyvar.ndim != 2 or ty.shape != tyvar.shape:
            raise ValueError((ty.shape, tyvar.shape))
    return tx, ty, tyvar


def _rustname(index_driver: NeighborIndex) -> str:
    if index_driver not in INDEX_NAMES:
        raise ValueError(f"Unsupported index driver: {index_driver}")
    name = INDEX_NAMES[index_driver]
    if name in ("cuda", "cuvs", "cuvs_cagra"):
        try:
            import importlib

            importlib.import_module("libcuvs").load_library()
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ):
            pass
    return name


def _resolvedriver(index_driver: NeighborIndex, device: Any) -> NeighborIndex:
    if device is None or index_driver != NeighborIndex.FLAT:
        return index_driver
    kind = getattr(device, "kind", str(device)).lower()
    if kind not in _DEVICE_DRIVERS:
        raise ValueError(f"Unsupported device: {device!r}")
    return _DEVICE_DRIVERS[kind]


def _resolveindex(
    index_driver: NeighborIndex, index: NeighborIndex | str | None, device: Any
) -> tuple[NeighborIndex, str | None]:
    if index is None:
        return _resolvedriver(index_driver, device), None
    if isinstance(index, NeighborIndex):
        return _resolvedriver(index, device), None
    key = str(index).lower()
    if key in _INDEX_ALIASES:
        return _resolvedriver(_INDEX_ALIASES[key], device), None
    return _resolvedriver(index_driver, device), str(index)


def _resolvestorage(
    index_driver: NeighborIndex,
    enn_storage: str | None,
    storage: Any,
    work_dir: str | os.PathLike[str] | None,
) -> str | None:
    if storage is not None:
        enn_storage = getattr(storage, "value", str(storage))
    has_dir = work_dir is not None
    if index_driver == NeighborIndex.BPANN_DISK and enn_storage is None and has_dir:
        return "disk"
    return enn_storage


@dataclass(frozen=True)
class ENNState:
    """Immutable value token representing an Epistemic Nearest Neighbors dataset & index."""

    _rust: Any
    scale_x: bool = False
    options: tuple[tuple[str, Any], ...] = ()
    _index_driver: NeighborIndex = NeighborIndex.FLAT
    _metric: tuple[str, int] = ("euclidean", 4)

    @property
    def x(self) -> np.ndarray:
        """Read-only observed inputs array."""
        return _readonly(self._rust.x_obs())

    @property
    def y(self) -> np.ndarray:
        """Read-only observed targets array."""
        return _readonly(self._rust.y_obs())

    @property
    def y_var(self) -> np.ndarray | None:
        """Read-only observation variances array, if present."""
        v = self._rust.yvar_obs()
        return None if v is None else _readonly(v)

    def __len__(self) -> int:
        return len(self._rust)

    @property
    def num_dim(self) -> int:
        """Return input feature dimension."""
        return int(self._rust.num_dim)

    @property
    def num_outputs(self) -> int:
        """Return number of output metrics."""
        return int(self._rust.num_outputs)

    @property
    def rust_backend(self) -> Any:
        """Rust surrogate implementation."""
        return self._rust

    @property
    def device(self) -> str:
        """Surrogate device name."""
        d = getattr(self._rust, "device", "cpu")
        return str(d() if callable(d) else d)

    @property
    def index_kind(self) -> str:
        """Kind of neighbor index backing this model."""
        if hasattr(self._rust, "index_kind"):
            k = self._rust.index_kind
            return str(k() if callable(k) else k)
        return _INDEX_KINDS.get(self._index_driver, "exact")

    @property
    def diagonal_metric(self) -> DiagonalMetric | None:
        """Diagonal metric parameters if metric learning is enabled."""
        dm = getattr(self._rust, "diagonal_metric", None)
        raw = dm() if callable(dm) else dm
        if raw is None:
            return None
        if isinstance(raw, DiagonalMetric):
            return raw
        if isinstance(raw, dict):
            return DiagonalMetric.from_dict(raw)
        return None

    @property
    def metric(self) -> DiagonalMetric | None:
        """Metric parameters if metric learning is enabled."""
        return self.diagonal_metric

    @property
    def y_bounds(self) -> np.ndarray:
        """Target value bounds per metric."""
        return np.asarray(self._rust.y_bounds, dtype=float)

    @property
    def index_stale(self) -> bool:
        """Whether the neighbor index has un-synchronized observations."""
        if hasattr(self._rust, "index_stale"):
            s = self._rust.index_stale
            return bool(s() if callable(s) else s)
        return False

    def memory_bytes(self) -> int:
        """Memory footprint of the neighbor index and model in bytes."""
        if hasattr(self._rust, "memory_bytes"):
            return int(self._rust.memory_bytes())
        if hasattr(self._rust, "index_bytes"):
            return int(self._rust.index_bytes())
        return 0

    def release_pages(self) -> Any:
        """Release unused memory pages back to the operating system."""
        if hasattr(self._rust, "release_pages"):
            return self._rust.release_pages()
        return None

    @property
    def key(self) -> Any:
        """Return content-addressed deterministic PRNG key for this state."""
        from .._rust import Key as _Key

        v = self.y_var
        buf = (
            np.ascontiguousarray(self.x, dtype=float).tobytes()
            + np.ascontiguousarray(self.y, dtype=float).tobytes()
            + (b"" if v is None else np.ascontiguousarray(v, dtype=float).tobytes())
            + repr(self.options).encode("utf-8")
        )
        return _Key.from_bytes(buf)

    def ensure_sync(self) -> ENNState:
        """Synchronize pending observations with the neighbor index."""
        return ensure_sync(self)

    def persist_index(self) -> ENNState:
        """Persist the neighbor index to storage."""
        return persist_index(self)

    def schedule_flush(self) -> ENNState:
        """Schedule a flush of pending observations to backing storage."""
        rust_model = self._rust.schedule_flush()
        return ENNState(
            _rust=rust_model,
            scale_x=self.scale_x,
            options=self.options,
            _index_driver=self._index_driver,
            _metric=self._metric,
        )

    def posterior(
        self,
        x: Any,
        params: ENNParams,
        *,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        """Compute posterior normal moments at query points."""
        flags = PosteriorFlags() if flags is None else flags
        mu, se, se_epi, se_ale, idx = self._rust.posterior(
            np.asarray(x, dtype=float),
            k_neighbors=int(params.k_neighbors),
            epistemic_scale=float(params.epistemic_scale),
            aleatoric_scale=float(params.aleatoric_scale),
            exclude_nearest=bool(flags.exclude_nearest),
            observation_noise=bool(flags.observation_noise),
            tie_mode=int(bool(flags.tie_neighbors)),
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=self.y_bounds)

    def posterior_draw(
        self,
        x: Any,
        params: ENNParams,
        *,
        function_seeds: Any,
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw deterministic posterior function samples at query points."""
        flags = PosteriorFlags() if flags is None else flags
        seeds = _rustseeds(function_seeds)
        draws, idx = self._rust.posterior_draw(
            np.asarray(x, dtype=float),
            k_neighbors=int(params.k_neighbors),
            epistemic_scale=float(params.epistemic_scale),
            aleatoric_scale=float(params.aleatoric_scale),
            function_seeds=seeds,
            exclude_nearest=bool(flags.exclude_nearest),
            observation_noise=bool(flags.observation_noise),
            tie_mode=int(bool(flags.tie_neighbors)),
        )
        return _finaldraw(np.asarray(x, dtype=float), draws, idx)

    def conditional_posterior(
        self,
        x_obs: Any,
        y_obs: Any,
        x_query: Any,
        params: ENNParams,
        *,
        yvar_obs: Any = None,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        """Compute posterior moments conditioned on hypothetical observations."""
        flags = PosteriorFlags() if flags is None else flags
        x_whatif = np.asarray(x_obs, dtype=float)
        y_whatif = np.asarray(y_obs, dtype=float)
        if y_whatif.ndim == 1:
            y_whatif = y_whatif.reshape(-1, 1)
        xq = np.asarray(x_query, dtype=float)
        kw: dict[str, Any] = {
            "k_neighbors": int(params.k_neighbors),
            "epistemic_scale": float(params.epistemic_scale),
            "aleatoric_scale": float(params.aleatoric_scale),
            "exclude_nearest": bool(flags.exclude_nearest),
            "observation_noise": bool(flags.observation_noise),
            "tie_mode": int(bool(flags.tie_neighbors)),
        }
        if yvar_obs is not None:
            yvar_arr = np.asarray(yvar_obs, dtype=float)
            if yvar_arr.ndim == 1:
                yvar_arr = yvar_arr.reshape(-1, 1)
            kw["yvar_whatif"] = yvar_arr
        try:
            mu, se, se_epi, se_ale, _ = self._rust.conditional_posterior(
                x_whatif, y_whatif, xq, **kw
            )
        except TypeError:
            kw.pop("yvar_whatif", None)
            mu, se, se_epi, se_ale, _ = self._rust.conditional_posterior(
                x_whatif, y_whatif, xq, **kw
            )
        return ENNNormal(mu, se, se_epi, se_ale, y_bounds=self.y_bounds)

    def conditional_draw(
        self,
        x_obs: Any,
        y_obs: Any,
        x_query: Any,
        params: ENNParams,
        *,
        function_seeds: Any,
        yvar_obs: Any = None,
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw posterior function samples conditioned on hypothetical observations."""
        flags = PosteriorFlags() if flags is None else flags
        x_whatif = np.asarray(x_obs, dtype=float)
        y_whatif = np.asarray(y_obs, dtype=float)
        if y_whatif.ndim == 1:
            y_whatif = y_whatif.reshape(-1, 1)
        xq = np.asarray(x_query, dtype=float)
        if x_whatif.shape[0] == 0:
            return self.posterior_draw(
                xq, params, function_seeds=function_seeds, flags=flags
            )
        seeds = _rustseeds(function_seeds)
        kw: dict[str, Any] = {
            "k_neighbors": int(params.k_neighbors),
            "epistemic_scale": float(params.epistemic_scale),
            "aleatoric_scale": float(params.aleatoric_scale),
            "function_seeds": seeds,
            "exclude_nearest": bool(flags.exclude_nearest),
            "observation_noise": bool(flags.observation_noise),
            "tie_mode": int(bool(flags.tie_neighbors)),
        }
        if yvar_obs is not None:
            yvar_arr = np.asarray(yvar_obs, dtype=float)
            if yvar_arr.ndim == 1:
                yvar_arr = yvar_arr.reshape(-1, 1)
            kw["yvar_whatif"] = yvar_arr
        try:
            draws, idx = self._rust.conditional_draw(x_whatif, y_whatif, xq, **kw)
        except TypeError:
            kw.pop("yvar_whatif", None)
            draws, idx = self._rust.conditional_draw(x_whatif, y_whatif, xq, **kw)
        return _finaldraw(xq, draws, idx)

    def nearest_neighbors(
        self,
        x: Any,
        *,
        search_k: int,
        exclude_nearest: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbor squared distances and indices."""
        dist2s, idx = self._rust.nearest_neighbors(
            np.asarray(x, dtype=float), int(search_k), bool(exclude_nearest)
        )
        return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)

    def neighbors(
        self,
        x: Any,
        k: int = 1,
        *,
        exclude_nearest: bool = False,
    ) -> np.ndarray:
        """Get k nearest neighbor indices for a 1D query point."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_mat = x_arr.reshape(1, -1)
        elif x_arr.ndim == 2:
            if x_arr.shape[0] != 1:
                raise ValueError(
                    f"2D query array must have shape (1, {self.num_dim}), got {x_arr.shape}"
                )
            x_mat = x_arr
        else:
            raise ValueError(
                f"query array must be 1D or 2D with batch size 1, got shape {x_arr.shape}"
            )
        if exclude_nearest and len(self) <= 1:
            raise ValueError(
                f"exclude_nearest=True requires at least 2 observations, got {len(self)}"
            )
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if x_mat.shape[1] != self.num_dim:
            raise ValueError(
                f"expected query with {self.num_dim} features, got {x_mat.shape[1]}"
            )
        if hasattr(self._rust, "neighbors"):
            out = self._rust.neighbors(x_mat, int(k), bool(exclude_nearest))
            return np.asarray(out[0], dtype=int)
        if len(self) == 0 or int(k) <= 0:
            return np.zeros(0, dtype=int)
        search_k = min(int(k) + int(exclude_nearest), len(self))
        _, idx = self.nearest_neighbors(
            x_mat, search_k=search_k, exclude_nearest=exclude_nearest
        )
        k_out = min(int(k), idx.shape[1])
        return np.asarray(idx[0, :k_out], dtype=int)

    def posterior_neighbors(
        self,
        x: Any,
        *,
        search_k: int,
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query posterior neighbor squared distances and indices."""
        flags = PosteriorFlags() if flags is None else flags
        dist2s, idx = self._rust.posterior_neighbors(
            np.asarray(x, dtype=float),
            int(search_k),
            bool(flags.exclude_nearest),
            bool(flags.tie_neighbors),
        )
        return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)

    def draw_internals(
        self,
        x: Any,
        params: ENNParams,
        *,
        function_seeds: Any,
        flags: PosteriorFlags | None = None,
    ) -> DrawInternals:
        """Compute intermediate neighbor weights and moments for an ENN draw."""
        if hasattr(self._rust, "draw_internals"):
            return self._rust.draw_internals(
                x, params, function_seeds=function_seeds, flags=flags
            )
        flags = PosteriorFlags() if flags is None else flags
        normal = self.posterior(x, params, flags=flags)
        dist2s, idx = self.posterior_neighbors(
            x, search_k=int(params.k_neighbors), flags=flags
        )
        eps = 1e-12
        inv_d = 1.0 / np.sqrt(np.maximum(dist2s, eps))
        sum_inv = np.sum(inv_d, axis=-1, keepdims=True)
        w = inv_d / np.maximum(sum_inv, eps)
        w_norm = np.repeat(w[:, :, np.newaxis], self.num_outputs, axis=2)
        l2 = np.sqrt(np.sum(w_norm**2, axis=1))
        return DrawInternals(
            idx=idx,
            w_normalized=w_norm,
            l2=l2,
            mu=normal.mu,
            se=normal.se,
            se_epi=normal.se_epi,
            se_ale=normal.se_ale,
        )

    def neighbor_data(
        self,
        x: Any,
        *,
        search_k: int,
        exclude_nearest: bool = False,
    ) -> NeighborData:
        """Query squared distances, indices, and target values for k nearest neighbors."""
        if hasattr(self._rust, "neighbor_data"):
            return self._rust.neighbor_data(
                x, search_k=search_k, exclude_nearest=exclude_nearest
            )
        dist2s, idx = self.nearest_neighbors(
            x, search_k=search_k, exclude_nearest=exclude_nearest
        )
        y_obs = self.y
        y_neighbors = y_obs[idx]
        return NeighborData(
            dist2s=dist2s,
            idx=idx,
            y_neighbors=y_neighbors,
            k=int(search_k),
        )

    def device_posterior(
        self,
        x: Any,
        params: ENNParams,
        *,
        flags: PosteriorFlags | None = None,
    ) -> Any:
        """Return a native posterior with DLPack mean, SE, and joint draw arrays."""
        if type(x).__module__.startswith(("jax.", "jaxlib.")):
            try:
                device, _ = x.__dlpack_device__()
            except BufferError as error:
                raise ValueError(
                    "JAX Metal arrays are not supported by device_posterior; "
                    "use Torch MPS or JAX CUDA"
                ) from error
        else:
            device, _ = x.__dlpack_device__()
        if int(device) == 8 and type(x).__module__.startswith("torch"):
            import torch

            torch.mps.synchronize()
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        flags = PosteriorFlags() if flags is None else flags
        return self._rust.device_posterior(
            x,
            k_neighbors=int(params.k_neighbors),
            epistemic_scale=float(params.epistemic_scale),
            aleatoric_scale=float(params.aleatoric_scale),
            exclude_nearest=bool(flags.exclude_nearest),
            observation_noise=bool(flags.observation_noise),
            tie_mode=int(bool(flags.tie_neighbors)),
        )

    def batch_posterior(
        self,
        x: Any,
        paramss: list[ENNParams],
        *,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        """Compute posterior moments across a batch of hyperparameter sets."""
        flags = PosteriorFlags() if flags is None else flags
        x_arr = np.asarray(x, dtype=float)
        if not paramss:
            raise ValueError("paramss must be non-empty")
        k_vals = [int(p.k_neighbors) for p in paramss]
        epi_scales = [float(p.epistemic_scale) for p in paramss]
        ale_scales = [float(p.aleatoric_scale) for p in paramss]
        mu, se, se_epi, se_ale = self._rust.batch_posterior(
            x_arr,
            k_values=k_vals,
            epistemic_scales=epi_scales,
            aleatoric_scales=ale_scales,
            exclude_nearest=bool(flags.exclude_nearest),
            observation_noise=bool(flags.observation_noise),
            tie_mode=int(bool(flags.tie_neighbors)),
        )
        return ENNNormal(mu, se, se_epi, se_ale, y_bounds=self.y_bounds)

    def train_rows(
        self, indices: list[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Fetch training inputs, targets, and noise variances for row indices."""
        idx = [int(i) for i in np.asarray(indices, dtype=int).ravel()]
        x, y, yvar = self._rust.train_rows(idx)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        yvar_arr = None if yvar is None else np.asarray(yvar, dtype=float)
        return x_arr, y_arr, yvar_arr

    def index_bytes(self) -> int:
        """Return memory footprint of the neighbor index in bytes."""
        return self.memory_bytes()


def with_geometry(model: Any, geometry: str, local_k: int = 4) -> Any:
    """Return a model or `ENNState` configured with a local-scale Riemannian dissimilarity geometry."""
    m = str(geometry).strip().lower()
    k = int(local_k)
    if m not in _METRICS or k < 1:
        raise ValueError(f"invalid geometry={geometry!r} or local_k={local_k}")
    if isinstance(model, ENNState):
        rust_obj = model._rust.append_cow(metric=m, local_k=k)
        opts = dict(model.options)
        opts["geometry"] = m
        opts["local_k"] = k
        return ENNState(
            _rust=rust_obj,
            scale_x=model.scale_x,
            options=tuple(sorted(opts.items())),
            _index_driver=model._index_driver,
            _metric=(m, k),
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
        p = fit(state, k=k_fit, num_candidates=20, num_samples=5)
        loo = posterior(state, state.x, p, flags=PosteriorFlags(exclude_nearest=True))
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
        rust_obj = target._rust.append_cow(stiefel=st_arr, affine=aff_tup)
        opts = dict(target.options)
        if st_arr is not None:
            opts["stiefel"] = st_arr.tobytes()
        if aff_tup is not None:
            opts["affine"] = aff_tup
        return ENNState(
            _rust=rust_obj,
            scale_x=target.scale_x,
            options=tuple(sorted(opts.items())),
            _index_driver=target._index_driver,
            _metric=target._metric,
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
    """Applies Stiefel subspace James-Stein contraction to a state, config, or Transform."""
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
    """Applies an Affine posterior location-scale map to a state, config, or Transform."""
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
    unknown = set(options) - {
        "index",
        "storage",
        "device",
        "metric_learning",
        "tied_dims",
        "unscaled_dims",
        "metric_seed",
        "config_path",
        "index_driver",
        "work_dir",
        "enn_storage",
        "y_bounds",
        "geometry",
        "local_k",
    }
    if unknown:
        raise TypeError(f"Unexpected keyword argument: {sorted(unknown)[0]!r}")
    x, y, yvar = _inputs(train_x, train_y, train_yvar)
    idx_driver = options.get("index_driver", NeighborIndex.FLAT)
    if isinstance(idx_driver, str):
        if idx_driver.lower() not in _INDEX_ALIASES:
            raise ValueError(f"Unknown index_driver: {idx_driver}")
        idx_driver = _INDEX_ALIASES[idx_driver.lower()]
    resolved_driver, raw_index = _resolveindex(
        idx_driver, options.get("index"), options.get("device")
    )
    work_dir = options.get("work_dir")
    enn_storage = _resolvestorage(
        resolved_driver, options.get("enn_storage"), options.get("storage"), work_dir
    )
    metric_spec = metric_options(options)
    config_path = options.get("config_path")
    y_bounds = options.get("y_bounds")

    rust_kwargs: dict[str, Any] = {
        "train_x": x,
        "train_y": y,
        "train_yvar": yvar,
        "scale_x": scale_x,
        "index_driver": _rustname(resolved_driver),
    }
    if raw_index is not None:
        rust_kwargs["index"] = raw_index
    if work_dir is not None:
        rust_kwargs["work_dir"] = os.fspath(work_dir)
    if enn_storage is not None:
        rust_kwargs["enn_storage"] = enn_storage
    if y_bounds is not None:
        rust_kwargs["y_bounds"] = np.asarray(y_bounds, dtype=float)
    if metric_spec is not None:
        rust_kwargs["metric"] = metric_spec
    if config_path is not None:
        rust_kwargs["config_path"] = os.fspath(config_path)

    rust_model = _RustENN(**rust_kwargs)
    st_arr, aff_tup = _stiefelarr(stiefel), _affinetup(affine)
    if st_arr is not None or aff_tup is not None:
        rust_model = rust_model.append_cow(stiefel=st_arr, affine=aff_tup)
    opts_tuple = tuple(
        sorted((k, v) for k, v in options.items() if not isinstance(v, np.ndarray))
    )
    return ENNState(
        _rust=rust_model,
        scale_x=scale_x,
        options=opts_tuple,
        _index_driver=resolved_driver,
        _metric=("euclidean", 4),
    )


def append(state: ENNState, x: Any, y: Any, y_var: Any = None) -> ENNState:
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
    x_in, y_in, yvar_in = _inputs(x_new, y_new, yvar_new)
    rust_model = state._rust.append_cow(x_in, y_in, yvar_in)
    return ENNState(
        _rust=rust_model,
        scale_x=state.scale_x,
        options=state.options,
        _index_driver=state._index_driver,
        _metric=state._metric,
    )


def posterior(
    state: ENNState, x: Any, params: ENNParams, *, flags: PosteriorFlags | None = None
) -> ENNNormal:
    """Pure posterior query on an immutable `ENNState`."""
    return state.posterior(x, params, flags=flags)


def sample(
    state: ENNState,
    x: Any,
    params: ENNParams,
    *,
    key: Any = None,
    num_draws: int = 1,
    seeds: Any = None,
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
    return state.posterior_draw(x, params, function_seeds=fn_seeds, flags=flags)


def fit(
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
        state._rust,
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


def ensure_sync(state: ENNState) -> ENNState:
    """Synchronize pending observations with the neighbor index, returning an updated state."""
    if isinstance(state, ENNState):
        rust_model = state._rust.ensure_sync()
        return ENNState(
            _rust=rust_model,
            scale_x=state.scale_x,
            options=state.options,
            _index_driver=state._index_driver,
            _metric=state._metric,
        )
    raise TypeError(f"Expected ENNState, got {type(state)}")


def persist_index(state: ENNState) -> ENNState:
    """Persist the neighbor index to storage, returning an updated state."""
    if isinstance(state, ENNState):
        rust_model = state._rust.persist_index()
        return ENNState(
            _rust=rust_model,
            scale_x=state.scale_x,
            options=state.options,
            _index_driver=state._index_driver,
            _metric=state._metric,
        )
    raise TypeError(f"Expected ENNState, got {type(state)}")


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
    "enn_init",
    "ensure_sync",
    "fit",
    "persist_index",
    "posterior",
    "sample",
    "stein",
    "with_geometry",
]
