"""OxCaml 5.2.0+ox 56-function benchmark suite via modal CPython C-API bridge.

Bridges baux's unboxed ``float#`` and ``float64x2#`` NEON/SLEEF SIMD benchmark
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
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_EXT: ModuleType | None = None
_JS_BRIDGE_CACHE: tuple[Any, _JsOxcamlBridge] | None = None


def _pyodide_ffi() -> tuple[Any, Any, Any]:
    try:
        from pyodide.ffi import JsException, create_proxy, to_js  # type: ignore[import-not-found]

        return create_proxy, to_js, JsException
    except ImportError:
        return None, None, None


def _to_py(val: Any) -> Any:
    return val.to_py() if hasattr(val, "to_py") else val


def _normalize_meta(raw: Any) -> dict[str, Any]:
    meta = dict(_to_py(raw))
    for key in ("id", "label", "family", "rule", "source"):
        if key in meta and meta[key] is not None:
            meta[key] = str(meta[key])
    for key in ("dim", "dimension", "default_dim", "min_dim", "dim_step"):
        if key in meta and meta[key] is not None:
            meta[key] = int(meta[key])
    if "scalable" in meta:
        meta["scalable"] = bool(meta["scalable"])
    for key in ("bounds", "domain"):
        if key in meta and meta[key] is not None:
            meta[key] = [(float(pair[0]), float(pair[1])) for pair in _to_py(meta[key])]
    return meta


class _JsOxcamlBridge:
    """Zero-copy Pyodide / Web Worker adapter around ``globalThis.BauxOxcaml``."""

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge

    def _fn(self, primary: str, fallback: str | None = None) -> Any:
        fn = getattr(self._bridge, primary, None)
        if fn is None and fallback is not None:
            fn = getattr(self._bridge, fallback, None)
        if fn is None:
            raise RuntimeError(f"OxCaml JS bridge is missing method {primary!r}")
        return fn

    def _call(self, fn: Any, *args: Any) -> Any:
        _, _, js_exc = _pyodide_ffi()
        try:
            return fn(*args)
        except Exception as exc:
            msg = str(exc)
            if "Unknown OxCaml benchmark" in msg:
                raise KeyError(msg) from exc
            if (js_exc is not None and isinstance(exc, js_exc)) or isinstance(
                exc, (KeyError, ValueError)
            ):
                if isinstance(exc, KeyError):
                    raise
                raise ValueError(msg) from exc
            raise

    def catalogue(self) -> list[dict[str, Any]]:
        raw = self._call(self._fn("catalogue"))
        return [_normalize_meta(item) for item in _to_py(raw)]

    def describe(self, name: str, dim: int = 0) -> dict[str, Any]:
        raw = self._call(self._fn("describe"), str(name), int(dim))
        return _normalize_meta(raw)

    def eval_one(self, name: str, point: np.ndarray) -> float:
        d = int(point.shape[0])
        fn = self._fn("evalOne", "eval_one")
        create_proxy, to_js, _ = _pyodide_ffi()
        if create_proxy is not None:
            mv = memoryview(point)
            proxy = create_proxy(mv)
            buf = None
            try:
                buf = proxy.getBuffer("f64")
                return float(self._call(fn, str(name), d, buf.data))
            finally:
                if buf is not None:
                    buf.release()
                proxy.destroy()
                mv.release()
        arg = to_js(point) if to_js is not None else point
        return float(self._call(fn, str(name), d, arg))

    def eval_batch_array(
        self,
        name: str,
        mat: np.ndarray,
        *,
        backend: str = "simd",
        workers: int = 1,
        block_pairs: int = 128,
    ) -> np.ndarray:
        del backend, workers, block_pairs
        n, d = int(mat.shape[0]), int(mat.shape[1])
        out = np.empty(n, dtype=np.float64)
        flat = mat.reshape(-1)
        fn = self._fn("evalBatch", "eval_batch")
        create_proxy, _, _ = _pyodide_ffi()
        if create_proxy is not None:
            in_mv = memoryview(flat)
            out_mv = memoryview(out)
            in_proxy = create_proxy(in_mv)
            out_proxy = create_proxy(out_mv)
            in_buf = None
            out_buf = None
            try:
                in_buf = in_proxy.getBuffer("f64")
                out_buf = out_proxy.getBuffer("f64")
                self._call(fn, str(name), d, n, in_buf.data, out_buf.data)
            finally:
                if out_buf is not None:
                    out_buf.release()
                if in_buf is not None:
                    in_buf.release()
                out_proxy.destroy()
                in_proxy.destroy()
                out_mv.release()
                in_mv.release()
            return out
        res = self._call(fn, str(name), d, n, flat, out)
        if res is not None and res is not out:
            arr = np.asarray(_to_py(res), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"evalBatch returned shape {arr.shape}, expected ({n},)"
                )
            out[:] = arr
        return out

    def eval_batch(
        self,
        name: str,
        points: np.ndarray,
        backend: str = "simd",
        workers: int = 1,
        block_pairs: int = 128,
    ) -> bytearray:
        out = self.eval_batch_array(
            name,
            points,
            backend=backend,
            workers=workers,
            block_pairs=block_pairs,
        )
        return bytearray(out.tobytes())

    def refcount_probe(self, _obj: object) -> dict[str, int]:
        return {"local_borrow_delta": 0, "unique_steal": 1}


def _resolve_js_bridge() -> _JsOxcamlBridge | None:
    global _JS_BRIDGE_CACHE
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return None
    gt = getattr(js, "globalThis", None)
    raw = None
    for host in (js, gt):
        if host is None:
            continue
        for attr in ("_baux_oxcaml", "BauxOxcaml"):
            cand = getattr(host, attr, None)
            if cand is not None:
                raw = cand
                break
        if raw is not None:
            break
    if raw is None:
        return None
    if _JS_BRIDGE_CACHE is not None and _JS_BRIDGE_CACHE[0] is raw:
        return _JS_BRIDGE_CACHE[1]
    adapter = _JsOxcamlBridge(raw)
    _JS_BRIDGE_CACHE = (raw, adapter)
    return adapter


def _load() -> Any:
    global _EXT

    if sys.platform == "emscripten":
        js_bridge = _resolve_js_bridge()
        if js_bridge is not None:
            return js_bridge
        if "_pyodide_core" in sys.modules or "pyodide" in sys.modules:
            raise RuntimeError(
                "OxCaml JS bridge (globalThis.BauxOxcaml / _baux_oxcaml) is not loaded"
            )

    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parents[3]
    override = os.environ.get("BAUX_OXCAML_DIR")
    ext_dir = Path(override) if override else (root / ".cache" / "baux" / "ocaml-pyext")
    try:
        ext_sfx = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    except Exception:
        ext_sfx = ".so"
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
        try:
            py_inc = sysconfig.get_path("include")
        except Exception:
            py_inc = None
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


def _ensure_batch_matrix(points: ArrayLike) -> np.ndarray:
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
    return mat


def raw_batch(
    name: str,
    points: ArrayLike,
    *,
    backend: str = "simd",
    workers: int = 1,
    block_pairs: int = 128,
) -> bytearray:
    """Evaluate a 2D ``(N, D)`` batch and return the stolen ``@ unique`` bytearray."""
    mat = _ensure_batch_matrix(points)
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
    mat = _ensure_batch_matrix(points)
    ext = _load()
    if hasattr(ext, "eval_batch_array"):
        return ext.eval_batch_array(
            name,
            mat,
            backend=backend,
            workers=int(workers),
            block_pairs=int(block_pairs),
        )
    raw = ext.eval_batch(name, mat, backend, int(workers), int(block_pairs))
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
    target_dim = int(bounds.shape[0])

    def objective(x: ArrayLike) -> float | np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != target_dim:
                raise ValueError(
                    f"Expected 1D point of length {target_dim}, got shape {arr.shape}"
                )
            val = eval_one(name, arr)
            return -val if negate else val
        if arr.ndim == 2:
            if arr.shape[1] != target_dim:
                raise ValueError(
                    f"Expected 2D batch with D={target_dim}, got shape {arr.shape}"
                )
            vals = eval_batch(name, arr, backend=backend, workers=workers)
            return -vals if negate else vals
        raise ValueError(f"Expected 1D point or 2D batch, got shape {arr.shape}")

    return bounds, objective
