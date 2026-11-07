"""PyTree-native parameter specifications, bounds, flattening, and MLP forward passes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LeafSpec:
    """Metadata and natural bounds for a single leaf in a parameter PyTree."""

    path: tuple[Any, ...]
    shape: tuple[int, ...]
    offset: int
    size: int
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class PyTreeSpec:
    """Immutable structural specification and natural bounds for a parameter PyTree."""

    dim: int
    bounds: np.ndarray
    leaves: tuple[LeafSpec, ...]
    template: Any = None

    def __post_init__(self) -> None:
        arr = np.array(self.bounds, dtype=float, copy=True)
        if arr.shape != (self.dim, 2):
            raise ValueError(f"bounds must have shape ({self.dim}, 2), got {arr.shape}")
        arr.flags.writeable = False
        object.__setattr__(self, "bounds", arr)

    def ravel(self, tree: Any) -> np.ndarray:
        """Flatten a parameter PyTree (or batch of PyTrees) into a 1D or 2D array."""
        return ravel(tree, self)

    def unflatten(self, flat: Any) -> Any:
        """Reconstruct a single parameter PyTree from a 1D flat vector."""
        return unflatten(flat, self)

    def unflatten_batch(self, flat_batch: Any) -> tuple[Any, ...]:
        """Reconstruct a tuple of parameter PyTrees from a 2D `(n, dim)` batch."""
        return unflatten_batch(flat_batch, self)

    def unflatten_stacked(self, flat_batch: Any) -> Any:
        """Reconstruct a single PyTree whose leaves have leading batch axis `(n, *shape)`."""
        return unflatten_stacked(flat_batch, self)


def _isnamedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(type(obj), "_fields")


def _iscontainer(obj: Any) -> bool:
    if isinstance(obj, (dict, Mapping, list)):
        return True
    if _isnamedtuple(obj):
        return True
    if is_dataclass(obj) and not isinstance(obj, type):
        return True
    if isinstance(obj, tuple):
        return not all(isinstance(v, int) for v in obj)
    return False


def _walkleaves(
    node: Any, path: tuple[Any, ...] = ()
) -> list[tuple[tuple[Any, ...], Any]]:
    if not _iscontainer(node):
        return [(path, node)]
    out: list[tuple[tuple[Any, ...], Any]] = []
    if isinstance(node, Mapping):
        for k in node:
            out.extend(_walkleaves(node[k], (*path, k)))
    elif _isnamedtuple(node):
        for k in node._fields:
            out.extend(_walkleaves(getattr(node, k), (*path, k)))
    elif is_dataclass(node) and not isinstance(node, type):
        for f in fields(node):
            out.extend(_walkleaves(getattr(node, f.name), (*path, f.name)))
    else:
        for i, item in enumerate(node):
            out.extend(_walkleaves(item, (*path, i)))
    return out


def _leafshape(leaf: Any) -> tuple[int, ...]:
    if isinstance(leaf, tuple) and all(isinstance(v, int) for v in leaf):
        if any(v <= 0 for v in leaf):
            raise ValueError(f"leaf shape dimensions must be > 0, got {leaf}")
        return tuple(int(v) for v in leaf)
    arr = np.asarray(leaf)
    return tuple(int(s) for s in arr.shape)


def _boundleaf(bound: Any, path: tuple[Any, ...], shape: tuple[int, ...]) -> np.ndarray:
    cur = bound
    for key in path:
        if not _iscontainer(cur):
            break
        if isinstance(cur, Mapping):
            cur = cur[key]
        elif _isnamedtuple(cur) or (is_dataclass(cur) and not isinstance(cur, type)):
            cur = getattr(cur, key)
        else:
            cur = cur[key]
    arr = np.broadcast_to(np.asarray(cur, dtype=float), shape).copy()
    arr.flags.writeable = False
    return arr


def pytree_bounds(tree: Any, low: Any = -1.0, high: Any = 1.0) -> PyTreeSpec:
    """Construct a `PyTreeSpec` from a template PyTree of arrays or shape tuples."""
    if isinstance(tree, PyTreeSpec):
        return tree
    entries = _walkleaves(tree)
    if not entries:
        raise ValueError("parameter PyTree must contain at least one leaf")
    specs: list[LeafSpec] = []
    bounds_parts: list[np.ndarray] = []
    offset = 0
    for path, leaf in entries:
        shape = _leafshape(leaf)
        size = int(np.prod(shape, dtype=int))
        lo = _boundleaf(low, path, shape)
        hi = _boundleaf(high, path, shape)
        if np.any(lo > hi):
            raise ValueError(f"low > high at leaf path {path}")
        specs.append(LeafSpec(path, shape, offset, size, lo, hi))
        bounds_parts.append(np.column_stack([lo.ravel(), hi.ravel()]))
        offset += size
    bounds = np.vstack(bounds_parts)
    bounds.flags.writeable = False
    return PyTreeSpec(dim=offset, bounds=bounds, leaves=tuple(specs), template=tree)


def mlp_spec(
    layers: Sequence[tuple[int, int]], *, low: float = -1.0, high: float = 1.0
) -> PyTreeSpec:
    """Build a `PyTreeSpec` for an MLP with `(out_dim, in_dim)` weight & `(out_dim,)` bias layers."""
    if not layers:
        raise ValueError("layers must be non-empty")
    template = tuple(
        {"weight": (int(out_dim), int(in_dim)), "bias": (int(out_dim),)}
        for out_dim, in_dim in layers
    )
    return pytree_bounds(template, low=low, high=high)


def _rebuild(
    template: Any, values: dict[tuple[Any, ...], Any], path: tuple[Any, ...] = ()
) -> Any:
    if not _iscontainer(template):
        return values[path]
    if isinstance(template, Mapping):
        return {k: _rebuild(template[k], values, (*path, k)) for k in template}
    if _isnamedtuple(template):
        cls = type(template)
        return cls(
            *(
                _rebuild(getattr(template, k), values, (*path, k))
                for k in template._fields
            )
        )
    if is_dataclass(template) and not isinstance(template, type):
        cls = type(template)
        return cls(
            **{
                f.name: _rebuild(getattr(template, f.name), values, (*path, f.name))
                for f in fields(template)
            }
        )
    items = [_rebuild(item, values, (*path, i)) for i, item in enumerate(template)]
    return tuple(items) if isinstance(template, tuple) else items


def unflatten(flat: Any, spec: PyTreeSpec) -> Any:
    """Reconstruct a single parameter PyTree from a 1D flat vector of length `spec.dim`."""
    if getattr(flat, "ndim", 0) != 1 or int(flat.shape[0]) != spec.dim:
        raise ValueError(
            f"flat vector must have shape ({spec.dim},), got {getattr(flat, 'shape', None)}"
        )
    values = {
        leaf.path: flat[leaf.offset : leaf.offset + leaf.size].reshape(leaf.shape)
        for leaf in spec.leaves
    }
    return _rebuild(spec.template, values)


def unflatten_stacked(flat_batch: Any, spec: PyTreeSpec) -> Any:
    """Reconstruct a single PyTree whose leaves have leading batch axis `(n, *leaf_shape)`."""
    if getattr(flat_batch, "ndim", 0) != 2 or int(flat_batch.shape[1]) != spec.dim:
        raise ValueError(
            f"flat_batch must have shape (n, {spec.dim}), got {getattr(flat_batch, 'shape', None)}"
        )
    n = int(flat_batch.shape[0])
    values = {
        leaf.path: flat_batch[:, leaf.offset : leaf.offset + leaf.size].reshape(
            (n, *leaf.shape)
        )
        for leaf in spec.leaves
    }
    return _rebuild(spec.template, values)


def unflatten_batch(flat_batch: Any, spec: PyTreeSpec) -> tuple[Any, ...]:
    """Reconstruct a tuple of `n` parameter PyTrees from a 2D `(n, spec.dim)` batch matrix."""
    arr = np.asarray(flat_batch, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != spec.dim:
        raise ValueError(f"flat_batch must have shape (n, {spec.dim}), got {arr.shape}")
    return tuple(unflatten(arr[i], spec) for i in range(arr.shape[0]))


def _ravelone(tree: Any, spec: PyTreeSpec | None) -> np.ndarray:
    entries = dict(_walkleaves(tree))
    if spec is None:
        parts = [np.asarray(v, dtype=float).ravel() for v in entries.values()]
        return np.concatenate(parts) if parts else np.empty((0,), dtype=float)
    parts = []
    for leaf in spec.leaves:
        if leaf.path not in entries:
            raise KeyError(f"missing PyTree leaf at path {leaf.path}")
        arr = np.asarray(entries[leaf.path], dtype=float)
        if arr.shape != leaf.shape:
            raise ValueError(
                f"leaf at {leaf.path} expected shape {leaf.shape}, got {arr.shape}"
            )
        parts.append(arr.ravel())
    return np.concatenate(parts)


def ravel(tree: Any, spec: PyTreeSpec | None = None) -> np.ndarray:
    """Flatten a single PyTree into `(dim,)` or a stacked/sequence batch into `(n, dim)`."""
    if isinstance(tree, np.ndarray):
        return np.asarray(tree, dtype=float)
    if spec is not None:
        entries = dict(_walkleaves(tree))
        first_leaf = spec.leaves[0]
        if first_leaf.path in entries:
            arr0 = np.asarray(entries[first_leaf.path], dtype=float)
            if arr0.shape == first_leaf.shape:
                return _ravelone(tree, spec)
            if (
                arr0.ndim == len(first_leaf.shape) + 1
                and arr0.shape[1:] == first_leaf.shape
            ):
                n = arr0.shape[0]
                parts = [
                    np.asarray(entries[leaf.path], dtype=float).reshape(n, leaf.size)
                    for leaf in spec.leaves
                ]
                return np.hstack(parts)
        if isinstance(tree, (list, tuple)):
            return np.vstack([_ravelone(item, spec) for item in tree])
    return _ravelone(tree, spec)


def mlp_forward(
    params: Sequence[Mapping[str, Any]],
    obs: Any,
    *,
    act: Callable[[Any], Any],
) -> Any:
    """Pure functional forward pass over a sequence of `{'weight': W, 'bias': b}` layers."""
    h = obs
    last = len(params) - 1
    for idx, layer in enumerate(params):
        h = layer["weight"] @ h + layer["bias"]
        if idx < last:
            h = act(h)
    return h


def flat_mlp(
    weights: Any,
    layers: Sequence[tuple[int, int]],
    obs: Any,
    *,
    act: Callable[[Any], Any],
) -> Any:
    """Pure functional forward pass unpacking a flat vector (or tuple of `(w, b)` pairs) over `layers`."""
    if isinstance(weights, (tuple, list)):
        pairs = tuple(weights)
    else:
        items = []
        offset = 0
        for fan_in, fan_out in layers:
            w_size = fan_in * fan_out
            w = weights[offset : offset + w_size].reshape((fan_in, fan_out))
            offset += w_size
            b = weights[offset : offset + fan_out]
            offset += fan_out
            items.append((w, b))
        pairs = tuple(items)
    h = obs
    last = len(pairs) - 1
    for idx, (w, b) in enumerate(pairs):
        h = h @ w + b
        if idx < last:
            h = act(h)
    return h


__all__ = [
    "LeafSpec",
    "PyTreeSpec",
    "flat_mlp",
    "mlp_forward",
    "mlp_spec",
    "pytree_bounds",
    "ravel",
    "unflatten",
    "unflatten_batch",
    "unflatten_stacked",
]
