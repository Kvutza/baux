"""OxCaml 5.2.0+ox 56-function benchmark suite via modal CPython C-API bridge.

Bridges BAUX's unboxed ``float#`` and ``float64x2#`` NEON/SLEEF SIMD benchmark
kernels into Python using OxCaml's ``local`` and ``unique`` modes for
compile-time reference-count safety (Trangez, Jane Street Tech Blog, 2026):

- Python arguments enter OxCaml as ``Py_obj.t @ local`` borrowed references
  with zero ``Py_INCREF`` / ``Py_DECREF`` traffic and cannot escape stack scope.
- Fresh Python return values leave OxCaml as ``Py_obj.t @ unique`` single-owner
  references (``ob_refcnt == 1``) stolen directly by CPython on return.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sysconfig
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_EXT: ModuleType | None = None


def _load() -> ModuleType:
    global _EXT
    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parents[3]
    override = os.environ.get("BAUX_OXCAML_DIR")
    ext_dir = Path(override) if override else (root / ".cache" / "baux" / "ocaml-pyext")
    ext_sfx = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = ext_dir / f"_baux_oxcaml{ext_sfx}"

    if not so_path.is_file():
        candidates = sorted(ext_dir.glob("_baux_oxcaml*.so"))
        if candidates:
            so_path = candidates[0]

    if not so_path.is_file():
        tool = root / "ocaml" / "tool.sh"
        if not tool.is_file():
            raise RuntimeError(
                f"Missing compiled OxCaml extension at {so_path} and build script {tool}"
            )
        env = os.environ.copy()
        py_inc = sysconfig.get_path("include")
        if py_inc:
            env.setdefault("BAUX_PYTHON_INCLUDE", py_inc)
        env.setdefault("BAUX_PYTHON_EXT_SUFFIX", ext_sfx)
        subprocess.run(
            [str(tool), "pyext", str(ext_dir)],
            cwd=str(root),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if not so_path.is_file():
            candidates = sorted(ext_dir.glob("_baux_oxcaml*.so"))
            if candidates:
                so_path = candidates[0]

    spec = importlib.util.spec_from_file_location("_baux_oxcaml", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT = mod
    return mod


def catalogue() -> list[dict[str, Any]]:
    """Return metadata dictionaries for all 56 OxCaml benchmark problems."""
    return list(_load().catalogue())


def describe(name: str, dim: int = 0) -> dict[str, Any]:
    """Return bounds and dimensionality metadata for ``name`` at ``dim`` (0 = default)."""
    return dict(_load().describe(str(name), int(dim)))


def eval_one(name: str, point: ArrayLike) -> float:
    """Evaluate minimization objective ``f(point)`` via unboxed ``float#``."""
    arr = np.ascontiguousarray(point, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"eval_one expects a 1D coordinate vector, got shape {arr.shape}"
        )
    return float(_load().eval_one(name, arr))


def raw_batch(
    name: str,
    points: ArrayLike,
    *,
    backend: str = "simd",
    workers: int = 1,
    block_pairs: int = 128,
) -> bytearray:
    """Evaluate a 2D ``(N, D)`` batch and return the stolen ``@ unique`` bytearray."""
    mat = (
        points
        if isinstance(points, np.ndarray)
        and points.ndim == 2
        and points.dtype == np.float64
        and points.flags.c_contiguous
        else np.ascontiguousarray(points, dtype=np.float64)
    )
    if mat.ndim != 2:
        raise ValueError(f"eval_batch expects a 2D (N, D) array, got shape {mat.shape}")
    return _load().eval_batch(name, mat, backend, int(workers), int(block_pairs))


def eval_batch(
    name: str,
    points: ArrayLike,
    *,
    backend: str = "simd",
    workers: int = 1,
    block_pairs: int = 128,
) -> np.ndarray:
    """Evaluate minimization objective ``f(X)`` on ``(N, D)`` batch with zero output copy."""
    raw = raw_batch(
        name,
        points,
        backend=backend,
        workers=workers,
        block_pairs=block_pairs,
    )
    return np.frombuffer(raw, dtype=np.float64)


def refcount_probe(obj: object) -> dict[str, int]:
    """Inspect ``@ local`` zero-INCREF borrow vs. ``with_release`` ``newref``/``decref``."""
    return dict(_load().refcount_probe(obj))


def problem(
    name: str,
    *,
    dim: int | None = None,
    negate: bool = True,
    backend: str = "simd",
    workers: int = 1,
) -> tuple[np.ndarray, Callable[[ArrayLike], float | np.ndarray]]:
    """Return ``(bounds, objective)`` ready for ``baux.init`` / ``baux.ask`` / ``baux.tell``.

    Because ``baux.tell`` maximizes rewards while benchmark formulas define minimization
    surfaces ``f(x)``, ``negate=True`` (the default) returns ``-f(x)``.
    """
    meta = describe(name, 0 if dim is None else int(dim))
    bounds = np.asarray(meta["bounds"], dtype=np.float64)

    def objective(x: ArrayLike) -> float | np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            val = eval_one(name, arr)
            return -val if negate else val
        if arr.ndim == 2:
            vals = eval_batch(name, arr, backend=backend, workers=workers)
            return -vals if negate else vals
        raise ValueError(f"Expected 1D point or 2D batch, got shape {arr.shape}")

    return bounds, objective
