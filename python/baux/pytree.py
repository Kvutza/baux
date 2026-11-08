"""PyTree-native parameter specifications, non-Cartesian manifold embeddings, and codomain maps.
References: Stein (1956); James & Stein (1961); Efron & Morris (1973); Higham (1986) (`Stiefel`);
Aitken (1935); Vovk et al. (2005); Lei et al. (2018) (`Affine`); Johnson & Lindenstrauss (1984);
Achlioptas (2003); Bhattacharyya (1943); Čencov (1982) (`Categorical`, `Periodic`, `Simplex`, `Log`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._embeddings import (
    Affine,
    Log,
    Periodic,
    Simplex,
    Stiefel,
    _catseed,
    _iscontainer,
    _isnamedtuple,
    _logwidth,
    _readonly,
    _rebuild,
    _spacekey,
    _walkleaves,
    argmax_dot,
    code_hash,
    flat_mlp,
    mlp_forward,
)


@dataclass(frozen=True)
class Categorical:
    """Isometric one-hot or Rademacher JL specification (Johnson & Lindenstrauss, 1984; Achlioptas, 2003)."""

    categories: tuple[Any, ...]
    encoding: str = "auto"
    seed: Any = None
    separation: float = 10.0
    dim_cap: int = 48
    key: Any = None

    def __post_init__(self) -> None:
        if not self.categories or self.encoding not in ("auto", "onehot", "rademacher"):
            raise ValueError("invalid categories or encoding")
        if not (
            1 <= self.dim_cap <= 64
            and np.isfinite(self.separation)
            and self.separation > 0.0
        ):
            raise ValueError("invalid dim_cap or separation")
        cats = tuple(self.categories)
        object.__setattr__(self, "categories", cats)
        object.__setattr__(self, "seed", _catseed(cats, self.seed, self.key))

    @property
    def card(self) -> int:
        return len(self.categories)

    @property
    def kind(self) -> str:
        cap = _logwidth(max(len(self.categories), 2), self.dim_cap)
        return (
            ("onehot" if len(self.categories) <= cap else "rademacher")
            if self.encoding == "auto"
            else self.encoding
        )

    @property
    def width(self) -> int:
        return (
            len(self.categories)
            if self.kind == "onehot"
            else _logwidth(len(self.categories), self.dim_cap)
        )

    @property
    def bounds(self) -> tuple[float, float]:
        s = float(self.separation) / float(np.sqrt(self.width))
        return (0.0, 1.0) if self.kind == "onehot" else (-s, s)

    def codebook(self, variable_idx: int = 0) -> np.ndarray:
        if self.kind == "onehot":
            return np.eye(len(self.categories), dtype=float)
        w, s = self.width, float(self.separation) / float(np.sqrt(self.width))
        bits = [
            [
                s if ((code_hash(i, variable_idx, self.seed) >> j) & 1) else -s
                for j in range(w)
            ]
            for i in range(len(self.categories))
        ]
        return np.array(bits, dtype=float)

    def encode(self, value: Any, variable_idx: int = 0) -> np.ndarray:
        m = [
            i
            for i, c in enumerate(self.categories)
            if type(c) is type(value) and c == value
        ]
        if not m and not isinstance(value, bool):
            m = [
                i
                for i, c in enumerate(self.categories)
                if not isinstance(c, bool) and c == value
            ]
        idx = (
            m[0]
            if m
            else (
                int(value)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                else -1
            )
        )
        if not (0 <= idx < len(self.categories)):
            raise ValueError(f"unknown category {value!r}")
        return self.codebook(variable_idx)[idx].copy()

    def snap(
        self, vec: Any, variable_idx: int = 0, weights: np.ndarray | None = None
    ) -> Any:
        return argmax_dot(
            np.asarray(vec, dtype=float),
            self.codebook(variable_idx),
            self.categories,
            weights=weights,
        )


_LEAFTYPES = (Categorical, Periodic, Simplex, Log)


def categorical(
    categories: Sequence[Any],
    *,
    encoding: str = "auto",
    key: Any = None,
    seed: Any = None,
    separation: float = 10.0,
    dim_cap: int = 48,
) -> Categorical:
    """Construct an isometric `Categorical` leaf specification."""
    return Categorical(
        tuple(categories),
        encoding=encoding,
        seed=seed,
        separation=separation,
        dim_cap=dim_cap,
        key=key,
    )


@dataclass(frozen=True)
class LeafSpec:
    """Metadata and natural bounds for a single leaf in a parameter PyTree."""

    path: tuple[Any, ...]
    shape: tuple[int, ...]
    offset: int
    size: int
    low: np.ndarray
    high: np.ndarray
    categorical: Categorical | None = None
    embed: Any = None


@dataclass(frozen=True, init=False)
class PyTreeSpec:
    """Immutable structural specification, manifold embeddings, and bounds for a parameter PyTree."""

    dim: int
    bounds: np.ndarray
    leaves: tuple[LeafSpec, ...]
    template: Any = None

    def __init__(
        self,
        dim: Any,
        bounds: Any = None,
        leaves: Any = None,
        template: Any = None,
        *,
        low: Any = -1.0,
        high: Any = 1.0,
    ) -> None:
        if leaves is None:
            built = pytree_bounds(dim, low=low if bounds is None else bounds, high=high)
            dim, bounds, leaves, template = (
                built.dim,
                built.bounds,
                built.leaves,
                built.template,
            )
        arr = _readonly(bounds)
        if arr.shape != (int(dim), 2):
            raise ValueError(f"bounds must have shape ({dim}, 2), got {arr.shape}")
        object.__setattr__(self, "dim", int(dim))
        object.__setattr__(self, "bounds", arr)
        object.__setattr__(self, "leaves", tuple(leaves))
        object.__setattr__(self, "template", template)

    @property
    def tied_dims(self) -> tuple[tuple[int, ...], ...]:
        """Column-index groups for multi-column manifold embedding leaves (`Categorical`, `Periodic`, `Simplex`)."""
        return tuple(
            tuple(range(s.offset, s.offset + s.size))
            for s in self.leaves
            if (s.categorical is not None or s.embed is not None) and s.size > 1
        )

    @property
    def key(self) -> Any:
        return _spacekey(self.bounds, self.leaves)

    def ravel(self, tree: Any) -> np.ndarray:
        return ravel(tree, self)

    def unflatten(self, flat: Any, weights: np.ndarray | None = None) -> Any:
        return unflatten(flat, self, weights=weights)

    def unflatten_batch(
        self, flat_batch: Any, weights: np.ndarray | None = None
    ) -> tuple[Any, ...]:
        return unflatten_batch(flat_batch, self, weights=weights)

    def unflatten_stacked(
        self, flat_batch: Any, weights: np.ndarray | None = None
    ) -> Any:
        return unflatten_stacked(flat_batch, self, weights=weights)

    flatten = ravel


Space = PyTreeSpec


def _boundleaf(bound: Any, path: tuple[Any, ...], shape: tuple[int, ...]) -> np.ndarray:
    cur = bound
    for key in path:
        if not _iscontainer(cur, _LEAFTYPES):
            break
        cur = (
            cur[key]
            if isinstance(cur, (Mapping, tuple, list)) and not _isnamedtuple(cur)
            else getattr(cur, key)
        )
    return _readonly(np.broadcast_to(np.asarray(cur, dtype=float), shape))


def _makespec(
    path: tuple[Any, ...], leaf: Any, offset: int, low: Any, high: Any
) -> LeafSpec:
    if isinstance(leaf, _LEAFTYPES):
        w, (lo_v, hi_v) = leaf.width, leaf.bounds
        cat = leaf if isinstance(leaf, Categorical) else None
        return LeafSpec(
            path,
            (w,),
            offset,
            w,
            _boundleaf(lo_v, (), (w,)),
            _boundleaf(hi_v, (), (w,)),
            cat,
            leaf,
        )
    if isinstance(leaf, tuple) and all(isinstance(v, int) for v in leaf):
        if any(v <= 0 for v in leaf):
            raise ValueError(f"leaf shape dimensions must be > 0, got {leaf}")
        shape = tuple(int(v) for v in leaf)
    else:
        shape = tuple(int(s) for s in np.asarray(leaf).shape)
    lo, hi = _boundleaf(low, path, shape), _boundleaf(high, path, shape)
    if np.any(lo > hi):
        raise ValueError(f"low > high at leaf path {path}")
    return LeafSpec(
        path, shape, offset, int(np.prod(shape, dtype=int)), lo, hi, None, None
    )


def pytree_bounds(tree: Any, low: Any = -1.0, high: Any = 1.0) -> PyTreeSpec:
    """Construct a `PyTreeSpec` (`Space`) from a template PyTree of arrays, shapes, or manifold embeddings."""
    if isinstance(tree, PyTreeSpec):
        return tree
    entries = _walkleaves(tree, _LEAFTYPES)
    if not entries:
        raise ValueError("parameter PyTree must contain at least one leaf")
    specs: list[LeafSpec] = []
    offset = 0
    for path, leaf in entries:
        spec = _makespec(path, leaf, offset, low, high)
        specs.append(spec)
        offset += spec.size
    bounds = np.vstack(
        [np.column_stack([s.low.ravel(), s.high.ravel()]) for s in specs]
    )
    return PyTreeSpec(offset, bounds, tuple(specs), tree)


def mlp_spec(
    layers: Sequence[tuple[int, int]], *, low: float = -1.0, high: float = 1.0
) -> PyTreeSpec:
    """Build a `PyTreeSpec` for an MLP with `(out_dim, in_dim)` weight & `(out_dim,)` bias layers."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return pytree_bounds(
        tuple({"weight": (int(o), int(i)), "bias": (int(o),)} for o, i in layers),
        low=low,
        high=high,
    )


def _decodeleaf(arr: Any, leaf: LeafSpec, weights: np.ndarray | None = None) -> Any:
    sub = arr[leaf.offset : leaf.offset + leaf.size]
    if leaf.categorical is not None:
        w_sub = None
        if weights is not None:
            w_arr = np.asarray(weights, dtype=float).ravel()
            w_sub = (
                w_arr[leaf.offset : leaf.offset + leaf.size]
                if w_arr.size != leaf.size
                else w_arr
            )
        return leaf.categorical.snap(sub, weights=w_sub)
    return leaf.embed.decode(sub) if leaf.embed is not None else sub.reshape(leaf.shape)


def unflatten(flat: Any, spec: PyTreeSpec, weights: np.ndarray | None = None) -> Any:
    """Reconstruct a single parameter PyTree from a 1D flat vector of length `spec.dim`."""
    arr = (
        flat
        if hasattr(flat, "ndim") and getattr(flat, "ndim", 0) > 0
        else np.atleast_1d(np.asarray(flat, dtype=float))
    )
    if getattr(arr, "ndim", 0) != 1 or int(arr.shape[0]) != spec.dim:
        raise ValueError(
            f"flat vector must have shape ({spec.dim},), got {getattr(arr, 'shape', None)}"
        )
    return _rebuild(
        spec.template,
        {leaf.path: _decodeleaf(arr, leaf, weights) for leaf in spec.leaves},
        _LEAFTYPES,
    )


def unflatten_stacked(
    flat_batch: Any, spec: PyTreeSpec, weights: np.ndarray | None = None
) -> Any:
    """Reconstruct a single PyTree whose leaves have leading batch axis `(n, *leaf_shape)`."""
    arr = (
        flat_batch
        if hasattr(flat_batch, "ndim")
        else np.asarray(flat_batch, dtype=float)
    )
    if getattr(arr, "ndim", 0) != 2 or int(arr.shape[1]) != spec.dim:
        raise ValueError(
            f"flat_batch must have shape (n, {spec.dim}), got {getattr(arr, 'shape', None)}"
        )
    n = int(arr.shape[0])
    values = {}
    for leaf in spec.leaves:
        sub = arr[:, leaf.offset : leaf.offset + leaf.size]
        if leaf.categorical is not None:
            values[leaf.path] = tuple(
                _decodeleaf(arr[i], leaf, weights) for i in range(n)
            )
        else:
            values[leaf.path] = (
                leaf.embed.decode(sub)
                if leaf.embed is not None
                else sub.reshape((n, *leaf.shape))
            )
    return _rebuild(spec.template, values, _LEAFTYPES)


def unflatten_batch(
    flat_batch: Any, spec: PyTreeSpec, weights: np.ndarray | None = None
) -> tuple[Any, ...]:
    """Reconstruct a tuple of `n` parameter PyTrees from a 2D `(n, spec.dim)` batch matrix."""
    arr = np.asarray(flat_batch, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != spec.dim:
        raise ValueError(f"flat_batch must have shape (n, {spec.dim}), got {arr.shape}")
    return tuple(unflatten(arr[i], spec, weights=weights) for i in range(arr.shape[0]))


def _encodeleaf(val: Any, leaf: LeafSpec) -> np.ndarray:
    if leaf.categorical is not None:
        if (
            isinstance(val, np.ndarray)
            and val.ndim == 1
            and val.dtype.kind not in ("i", "u")
        ):
            return np.vstack([leaf.categorical.encode(item) for item in val])
        return leaf.categorical.encode(val)
    if leaf.embed is not None:
        return leaf.embed.encode(val)
    arr = np.asarray(val, dtype=float)
    if arr.shape == leaf.shape:
        return arr.ravel()
    if arr.ndim == len(leaf.shape) + 1 and arr.shape[1:] == leaf.shape:
        return arr.reshape(arr.shape[0], leaf.size)
    raise ValueError(
        f"leaf at {leaf.path} expected shape {leaf.shape}, got {arr.shape}"
    )


def _ravelone(tree: Any, spec: PyTreeSpec | None) -> np.ndarray:
    entries = dict(_walkleaves(tree, _LEAFTYPES))
    if spec is None:
        parts = [np.asarray(v, dtype=float).ravel() for v in entries.values()]
        return np.concatenate(parts) if parts else np.empty((0,), dtype=float)
    for leaf in spec.leaves:
        if leaf.path not in entries:
            raise KeyError(f"missing PyTree leaf at path {leaf.path}")
    return np.concatenate(
        [_encodeleaf(entries[leaf.path], leaf) for leaf in spec.leaves]
    )


def ravel(tree: Any, spec: PyTreeSpec | None = None) -> np.ndarray:
    """Flatten a single PyTree into `(dim,)` or a stacked/sequence batch into `(n, dim)`."""
    if isinstance(tree, np.ndarray) and (
        spec is None
        or (spec.leaves[0].categorical is None and spec.leaves[0].embed is None)
    ):
        return np.asarray(tree, dtype=float)
    if spec is not None:
        entries = dict(_walkleaves(tree, _LEAFTYPES))
        first_leaf = spec.leaves[0]
        if first_leaf.path in entries:
            enc0 = _encodeleaf(entries[first_leaf.path], first_leaf)
            if enc0.ndim == 1:
                return _ravelone(tree, spec)
            n = enc0.shape[0]
            return np.hstack(
                [
                    _encodeleaf(entries[leaf.path], leaf).reshape(n, leaf.size)
                    for leaf in spec.leaves
                ]
            )
        if isinstance(tree, (list, tuple)):
            return np.vstack([_ravelone(item, spec) for item in tree])
    return _ravelone(tree, spec)


__all__ = [
    "Affine",
    "Categorical",
    "LeafSpec",
    "Log",
    "Periodic",
    "PyTreeSpec",
    "Simplex",
    "Space",
    "Stiefel",
    "argmax_dot",
    "categorical",
    "code_hash",
    "flat_mlp",
    "mlp_forward",
    "mlp_spec",
    "pytree_bounds",
    "ravel",
    "unflatten",
    "unflatten_batch",
    "unflatten_stacked",
]
