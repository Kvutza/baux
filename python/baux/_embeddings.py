"""Riemannian manifold embeddings (`Periodic`, `Simplex`, `Log`), `Stiefel`, and `Affine`.

References:
- Stein (1956), Proc. Third Berkeley Symp. 1:197-206; James & Stein (1961), Proc.
  Fourth Berkeley Symp. 1:361-379; Efron & Morris (1973), JASA 68(341):117-130;
  Higham (1986), SIAM J. Sci. Stat. Comput. 7(4):1160-1174 (`Stiefel`).
- Aitken (1935), Proc. R. Soc. Edinburgh 55:42-48; Vovk et al. (2005), Algorithmic
  Learning in a Random World; Lei et al. (1918/2018), JASA 113(523):1094-1111,
  arXiv:1604.04173 (`Affine`).
- Bhattacharyya (1943), Bull. Calcutta Math. Soc. 35:99-109; Čencov (1982),
  Statistical Decision Rules and Optimal Inference (`Periodic`, `Simplex`, `Log`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any

import numpy as np

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _readonly(arr: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _as2d(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def code_hash(cat_idx: int, variable_idx: int = 0, seed: int = 0) -> int:
    """SplitMix64 category hash matching `categorical.rs`."""
    x = (
        (cat_idx & _MASK64)
        ^ (seed & _MASK64)
        ^ ((variable_idx * 0x9E3779B97F4A7C15) & _MASK64)
    ) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _catseed(categories: tuple[Any, ...], seed: Any = None, key: Any = None) -> int:
    src = key if key is not None else seed
    if src is None:
        h = 0xCBF29CE484222325
        for b in repr(categories).encode("utf-8"):
            h = code_hash(b, 0, h)
        return h
    if hasattr(src, "key"):
        src = src.key
    if hasattr(src, "seed") and callable(src.seed):
        return int(src.seed()) & _MASK64
    return int(src) & _MASK64


def _spacekey(bounds: np.ndarray, leaves: tuple[Any, ...]) -> Any:
    from . import _rust

    payload = np.ascontiguousarray(bounds, dtype=float).tobytes() + repr(
        tuple((s.path, s.shape, s.categorical, s.embed) for s in leaves)
    ).encode("utf-8")
    return _rust.Key.from_bytes(payload)


def _logwidth(card: int, dim_cap: int) -> int:
    return (
        1 if card <= 1 else min(int(np.ceil(2.0 * np.log2(card) + 7.0)), int(dim_cap))
    )


def argmax_dot(
    vec: np.ndarray,
    codebook: np.ndarray,
    categories: tuple[Any, ...],
    weights: np.ndarray | None = None,
) -> Any:
    """Snap a continuous code vector via (Riemannian-weighted) maximum inner product."""
    v = np.asarray(vec, dtype=float).ravel()
    if weights is not None:
        v = v * np.asarray(weights, dtype=float).ravel()
    return categories[int(np.argmax(codebook @ v))]


def _fitcal(
    mu: Any, y: Any, se: Any, *, weighted: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m2, y2 = _as2d(mu), _as2d(y)
    if m2.ndim != 2 or m2.shape != y2.shape or m2.shape[0] == 0:
        raise ValueError(f"mu {m2.shape} and y {y2.shape} must match (N, M)")
    s2 = None if se is None else np.maximum(_as2d(se), 1e-12)
    if s2 is not None and s2.shape != m2.shape:
        raise ValueError(f"se {s2.shape} must match mu {m2.shape}")
    w = np.ones_like(m2) if (not weighted or s2 is None) else 1.0 / (s2 * s2)
    sw, smu, sy = np.sum(w, axis=0), np.sum(w * m2, axis=0), np.sum(w * y2, axis=0)
    smumu, smuy = np.sum(w * m2 * m2, axis=0), np.sum(w * m2 * y2, axis=0)
    det = sw * smumu - smu * smu
    ok = (m2.shape[0] >= 2) & (
        np.abs(det) >= 1e-18 * np.maximum(np.abs(sw * smumu), 1.0)
    )
    sdet = np.where(ok, det, 1.0)
    a = np.where(ok, (smumu * sy - smu * smuy) / sdet, 0.0)
    b = np.where(ok, (sw * smuy - smu * sy) / sdet, 1.0)
    if s2 is None:
        return a, b, np.ones(m2.shape[1], dtype=float)
    z = (y2 - (a[None, :] + b[None, :] * m2)) / s2
    c = np.sqrt(np.mean(z * z, axis=0))
    return a, b, np.where(np.isfinite(c) & (c > 1e-6), c, 1.0)


@dataclass(frozen=True)
class Periodic:
    """Isometric S^1 circle embedding mapping theta in [low, high) to 2 tied coordinates in [0, 1]^2."""

    low: float = 0.0
    high: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        lo, hi = float(self.low), float(self.high)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError("Periodic requires finite low < high")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    @property
    def width(self) -> int:
        return 2

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def encode(self, theta: Any) -> np.ndarray:
        arr = np.asarray(theta, dtype=float)
        ang = 2.0 * np.pi * (arr - self.low) / (self.high - self.low)
        c0, c1 = 0.5 + np.cos(ang) / (2.0 * np.pi), 0.5 + np.sin(ang) / (2.0 * np.pi)
        return (
            np.array([float(c0), float(c1)])
            if arr.ndim == 0
            else np.column_stack([c0.ravel(), c1.ravel()])
        )

    def decode(self, u: Any) -> Any:
        arr = np.asarray(u, dtype=float)
        if arr.ndim == 1:
            ang = float(np.mod(np.arctan2(arr[1] - 0.5, arr[0] - 0.5), 2.0 * np.pi))
            return float(self.low + (ang / (2.0 * np.pi)) * (self.high - self.low))
        ang = np.mod(np.arctan2(arr[:, 1] - 0.5, arr[:, 0] - 0.5), 2.0 * np.pi)
        return self.low + (ang / (2.0 * np.pi)) * (self.high - self.low)

    flatten = encode
    unflatten = decode
    snap = decode


@dataclass(frozen=True)
class Simplex:
    """Fisher-Rao Hellinger square-root embedding psi_i(p) = sqrt(p_i) on Delta^{d-1} (Bhattacharyya, 1943; Čencov, 1982)."""

    dim: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.dim, bool)
            or not isinstance(self.dim, (int, np.integer))
            or int(self.dim) < 1
        ):
            raise ValueError(f"Simplex.dim must be an integer >= 1, got {self.dim!r}")
        object.__setattr__(self, "dim", int(self.dim))

    @property
    def width(self) -> int:
        return self.dim

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def encode(self, p: Any) -> np.ndarray:
        arr = np.asarray(p, dtype=float)
        if arr.shape[-1:] != (self.dim,):
            raise ValueError(f"Simplex expected last axis {self.dim}, got {arr.shape}")
        return np.sqrt(np.maximum(arr, 0.0))

    def decode(self, u: Any) -> np.ndarray:
        arr = np.asarray(u, dtype=float)
        w = np.maximum(arr, 0.0) ** 2
        if arr.ndim == 1:
            s = float(np.sum(w))
            return w / s if s > 1e-30 else np.full(self.dim, 1.0 / self.dim)
        s = np.sum(w, axis=-1, keepdims=True)
        return np.where(s > 1e-30, w / np.maximum(s, 1e-30), 1.0 / self.dim)

    flatten = encode
    unflatten = decode
    snap = decode


@dataclass(frozen=True)
class Log:
    """Scale-invariant multiplicative Riemannian embedding on [low, high] subset R_+ (Čencov, 1982)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        lo, hi = float(self.low), float(self.high)
        if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
            raise ValueError("Log requires finite 0 < low < high")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    @property
    def width(self) -> int:
        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def encode(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if np.any(arr <= 0.0):
            raise ValueError("Log.encode requires strictly positive inputs")
        u = np.log(arr / self.low) / np.log(self.high / self.low)
        return np.array([float(u)], dtype=float) if arr.ndim == 0 else u.reshape(-1, 1)

    def decode(self, u: Any) -> Any:
        arr = np.asarray(u, dtype=float)
        ratio = self.high / self.low
        if arr.ndim <= 1 and arr.size == 1:
            return float(self.low * (ratio ** float(arr.reshape(-1)[0])))
        return self.low * (ratio ** arr.reshape(-1))

    flatten = encode
    unflatten = decode
    snap = decode


@dataclass(frozen=True)
class Stiefel:
    """Orthonormal subspace frame `(M, F)` on the Stiefel manifold `V_F(R^M)` for James-Stein contraction (Stein, 1956; James & Stein, 1961; Efron & Morris, 1973; Higham, 1986)."""

    loadings: np.ndarray

    def __post_init__(self) -> None:
        arr = _readonly(self.loadings)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Stiefel.loadings must be non-empty 2D, got {arr.shape}")
        object.__setattr__(self, "loadings", arr)

    @classmethod
    def from_factors(cls, factors: np.ndarray) -> Stiefel:
        """Project a full-column-rank factor matrix onto the Stiefel manifold V_F(R^M) (Higham, 1986)."""
        arr = np.asarray(factors, dtype=float)
        if (
            arr.ndim != 2
            or arr.shape[0] < arr.shape[1]
            or arr.shape[1] == 0
            or not np.all(np.isfinite(arr))
        ):
            raise ValueError("factors must be finite (M, F) with M >= F >= 1")
        u, s, vh = np.linalg.svd(arr, full_matrices=False)
        if np.any(s <= 1e-12):
            raise ValueError("factors must have full column rank")
        return cls(u @ vh)

    @classmethod
    def from_cov(cls, y: np.ndarray, rank: int = 1) -> Stiefel:
        """Extract the top `rank` principal directions of `(N, M)` observations onto `V_rank(R^M)`."""
        arr = np.asarray(y, dtype=float)
        r = int(rank)
        if (
            arr.ndim != 2
            or r < 1
            or r > arr.shape[1]
            or arr.shape[0] < r
            or not np.all(np.isfinite(arr))
        ):
            raise ValueError("y must be finite (N, M) with min(N, M) >= rank >= 1")
        yc = arr - np.mean(arr, axis=0, keepdims=True)
        _, s, vh = np.linalg.svd(yc, full_matrices=False)
        if np.any(s[:r] <= 1e-12):
            raise ValueError("y covariance has rank < requested rank")
        return cls(vh[:r].T)

    def transform_stein(
        self, mu: np.ndarray, se: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive Empirical-Bayes James-Stein orthogonal-subspace contraction per row."""
        mu_in, se_in = np.asarray(mu, dtype=float), np.asarray(se, dtype=float)
        mu_2d = mu_in.reshape(1, -1) if mu_in.ndim == 1 else mu_in
        se_2d = se_in.reshape(1, -1) if se_in.ndim == 1 else se_in
        if mu_2d.shape != se_2d.shape or mu_2d.shape[1] != self.loadings.shape[0]:
            raise ValueError(f"mu {mu_2d.shape} and se {se_2d.shape} shape mismatch")
        proj = self.loadings @ self.loadings.T
        p_diag = np.clip(np.diag(proj), 0.0, 1.0)
        d_perp = np.maximum(0.0, 1.0 - p_diag)
        mu_par = mu_2d @ proj.T
        r_perp = mu_2d - mu_par
        r_norm2 = np.sum(r_perp * r_perp, axis=1, keepdims=True)
        denom = r_norm2 + np.sum(
            (se_2d * se_2d) * d_perp[None, :], axis=1, keepdims=True
        )
        alpha = np.where(denom > 1e-30, r_norm2 / np.maximum(denom, 1e-30), 0.0)
        mu_out = mu_par + alpha * r_perp
        se_out = se_2d * np.sqrt(p_diag[None, :] + (alpha * alpha) * d_perp[None, :])
        if mu_in.ndim == 1:
            return _readonly(mu_out.reshape(-1)), _readonly(se_out.reshape(-1))
        return _readonly(mu_out), _readonly(se_out)

    stein = transform_stein


@dataclass(frozen=True)
class Affine:
    """Affine location-scale posterior map `(a + b*mu, c*se) in Aff(1)^M` (Aitken, 1935; Vovk et al., 2005; Lei et al., 2018)."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        a_arr, b_arr, c_arr = (
            _readonly(self.a).reshape(-1),
            _readonly(self.b).reshape(-1),
            _readonly(self.c).reshape(-1),
        )
        if not (a_arr.shape == b_arr.shape == c_arr.shape):
            raise ValueError("Affine a, b, c must have matching lengths")
        object.__setattr__(self, "a", a_arr)
        object.__setattr__(self, "b", b_arr)
        object.__setattr__(self, "c", c_arr)

    @classmethod
    def fit(cls, mu: np.ndarray, y: np.ndarray, se: np.ndarray | None = None) -> Affine:
        """Fit unweighted OLS `(a, b)` and residual uncertainty dispersion multiplier `c`."""
        return cls(*_fitcal(mu, y, se, weighted=False))

    @classmethod
    def fit_wls(cls, mu: np.ndarray, y: np.ndarray, se: np.ndarray) -> Affine:
        """Fit inverse-variance Weighted Least Squares (`w_i = se_i^-2`) `(a, b)` and dispersion `c`."""
        return cls(*_fitcal(mu, y, se, weighted=True))

    def conformal_scale(
        self, mu: np.ndarray, y: np.ndarray, se: np.ndarray, alpha: float = 0.1
    ) -> np.ndarray:
        """Compute finite-sample conformal quantile multiplier `q_{1-alpha}` on `|y - (a + b*mu)| / se`."""
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        m2, y2, s2 = _as2d(mu), _as2d(y), np.maximum(_as2d(se), 1e-12)
        if not (m2.shape == y2.shape == s2.shape and m2.shape[1] == self.a.shape[0]):
            raise ValueError("mu, y, se shapes must match (N, M)")
        n = m2.shape[0]
        z = np.sort(np.abs(y2 - (self.a[None, :] + self.b[None, :] * m2)) / s2, axis=0)
        idx = min(max(int(np.ceil((n + 1) * (1.0 - float(alpha)))) - 1, 0), n - 1)
        return _readonly(z[idx])


def _isnamedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(type(obj), "_fields")


def _iscontainer(obj: Any, leaf_types: tuple[type, ...]) -> bool:
    if isinstance(obj, leaf_types):
        return False
    if isinstance(obj, (dict, Mapping, list)) or _isnamedtuple(obj):
        return True
    if is_dataclass(obj) and not isinstance(obj, type):
        return True
    return isinstance(obj, tuple) and not all(isinstance(v, int) for v in obj)


def _walkleaves(
    node: Any, leaf_types: tuple[type, ...], path: tuple[Any, ...] = ()
) -> list[tuple[tuple[Any, ...], Any]]:
    if not _iscontainer(node, leaf_types):
        return [(path, node)]
    if isinstance(node, Mapping):
        pairs = [(k, node[k]) for k in node]
    elif _isnamedtuple(node):
        pairs = [(k, getattr(node, k)) for k in node._fields]
    elif is_dataclass(node) and not isinstance(node, type):
        pairs = [(f.name, getattr(node, f.name)) for f in fields(node)]
    else:
        pairs = list(enumerate(node))
    return [item for k, v in pairs for item in _walkleaves(v, leaf_types, (*path, k))]


def _rebuild(
    template: Any,
    values: dict[tuple[Any, ...], Any],
    leaf_types: tuple[type, ...],
    path: tuple[Any, ...] = (),
) -> Any:
    if not _iscontainer(template, leaf_types):
        return values[path]
    if isinstance(template, Mapping):
        return {
            k: _rebuild(template[k], values, leaf_types, (*path, k)) for k in template
        }
    if _isnamedtuple(template):
        return type(template)(
            *(
                _rebuild(getattr(template, k), values, leaf_types, (*path, k))
                for k in template._fields
            )
        )
    if is_dataclass(template) and not isinstance(template, type):
        return type(template)(
            **{
                f.name: _rebuild(
                    getattr(template, f.name), values, leaf_types, (*path, f.name)
                )
                for f in fields(template)
            }
        )
    items = [
        _rebuild(item, values, leaf_types, (*path, i))
        for i, item in enumerate(template)
    ]
    return tuple(items) if isinstance(template, tuple) else items


def mlp_forward(
    params: Sequence[Mapping[str, Any]], obs: Any, *, act: Callable[[Any], Any]
) -> Any:
    """Pure functional forward pass over a sequence of `{'weight': W, 'bias': b}` layers."""
    h = obs
    for idx, layer in enumerate(params):
        h = layer["weight"] @ h + layer["bias"]
        if idx < len(params) - 1:
            h = act(h)
    return h


def flat_mlp(
    weights: Any,
    layers: Sequence[tuple[int, int]],
    obs: Any,
    *,
    act: Callable[[Any], Any],
) -> Any:
    """Pure functional forward pass unpacking a flat vector over `layers`."""
    if not isinstance(weights, (tuple, list)):
        offsets = np.cumsum([0] + [(fi + 1) * fo for fi, fo in layers[:-1]])
        weights = tuple(
            (
                weights[s : s + fi * fo].reshape((fi, fo)),
                weights[s + fi * fo : s + (fi + 1) * fo],
            )
            for (fi, fo), s in zip(layers, offsets)
        )
    h = obs
    for idx, (w, b) in enumerate(weights):
        h = h @ w + b
        if idx < len(weights) - 1:
            h = act(h)
    return h
