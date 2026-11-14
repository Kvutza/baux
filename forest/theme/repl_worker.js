/* Web Worker hosting self-hosted stripped Pyodide + verified baux_py.zip */
"use strict";

importScripts("./pyodide/pyodide.js");

let runtimePromise = null;

function bytesToHex(buffer) {
  const bytes = new Uint8Array(buffer);
  const parts = new Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) {
    parts[i] = bytes[i].toString(16).padStart(2, "0");
  }
  return parts.join("");
}

async function fetchJsonStrict(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error("HTTP " + resp.status + " fetching " + url);
  }
  return await resp.json();
}

async function fetchVerifiedBytes(url, expectedSize, expectedSha256) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error("HTTP " + resp.status + " fetching " + url);
  }
  const buf = await resp.arrayBuffer();
  if (buf.byteLength !== expectedSize) {
    throw new Error(
      "Size mismatch for " + url + ": got " + buf.byteLength + ", expected " + expectedSize
    );
  }
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const actualHex = bytesToHex(digest);
  if (actualHex !== expectedSha256.toLowerCase()) {
    throw new Error(
      "SHA-256 mismatch for " + url + ": got " + actualHex + ", expected " + expectedSha256
    );
  }
  return buf;
}

const BOOTSTRAP_PY = `
import ast
import code
import io
import json
import math
import rlcompleter
import sys
import time
import traceback

_PKG_DIR = "/home/pyodide/baux_pkg"
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import numpy as np
import baux
from baux.benchmarks import oxcaml


def _make_ns():
    return {
        "__name__": "__main__",
        "np": np,
        "numpy": np,
        "baux": baux,
        "oxcaml": oxcaml,
    }


_REPL_NS = _make_ns()
_CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _reset_namespace():
    global _REPL_NS, _CONSOLE
    _REPL_NS = _make_ns()
    _CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _json_scalar(val):
    if isinstance(val, (bool, int)):
        return int(val) if not isinstance(val, bool) else val
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _clean_float_list(seq, limit=128):
    out = []
    try:
        arr = np.asarray(seq, dtype=float).ravel().tolist()
    except Exception:
        return out
    for v in arr[:limit]:
        fv = _json_scalar(v)
        if fv is not None:
            out.append(round(fv, 6))
    return out


def _clean_pairs(seq, limit=128):
    out = []
    try:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return out
        for row in arr[:limit]:
            rx = _json_scalar(row[0])
            ry = _json_scalar(row[1])
            if rx is not None and ry is not None:
                out.append([round(rx, 5), round(ry, 5)])
    except Exception:
        return out
    return out


def _sanitize_curves(raw):
    if not isinstance(raw, dict):
        return None
    raw_series = raw.get("series")
    if not isinstance(raw_series, (list, tuple)) or not raw_series:
        return None
    series_out = []
    for item in raw_series[:6]:
        if not isinstance(item, dict):
            continue
        ys = _clean_float_list(item.get("y", []), limit=160)
        if not ys:
            continue
        xs = _clean_float_list(item.get("x", []), limit=160)
        if len(xs) != len(ys):
            xs = list(range(1, len(ys) + 1))
        series_out.append({
            "label": str(item.get("label", f"series {len(series_out) + 1}"))[:40],
            "x": xs,
            "y": ys,
            "dashed": bool(item.get("dashed", False)),
        })
    if not series_out:
        return None
    return {
        "kind": "curves",
        "title": str(raw.get("title", "Comparison"))[:80],
        "subtitle": str(raw.get("subtitle", ""))[:120],
        "x_label": str(raw.get("x_label", "Evaluations"))[:40],
        "y_label": str(raw.get("y_label", "Objective"))[:40],
        "log_y": bool(raw.get("log_y", False)),
        "series": series_out,
    }


def _extract_fig(raw):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", ""))
    if kind == "curves":
        return _sanitize_curves(raw)

    if kind == "tokamak":
        fg = raw.get("flux_grid") or {}
        nr = int(fg.get("nr", 0)) if isinstance(fg, dict) else 0
        nz = int(fg.get("nz", 0)) if isinstance(fg, dict) else 0
        if nr <= 1 or nz <= 1 or nr * nz > 2500:
            return None
        coils_out = []
        for c in (raw.get("coils") or [])[:40]:
            try:
                coils_out.append([round(float(x), 4) for x in c[:5]])
            except Exception:
                pass
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        out = {
            "kind": "tokamak",
            "title": str(raw.get("title", "Tokamak Equilibrium"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "r_range": _clean_float_list(raw.get("r_range", [0.4, 3.0]), limit=2),
            "z_range": _clean_float_list(raw.get("z_range", [-1.6, 1.6]), limit=2),
            "first_wall": _clean_pairs(raw.get("first_wall", []), limit=128),
            "lcfs": _clean_pairs(raw.get("lcfs", []), limit=128),
            "x_point": _clean_float_list(raw.get("x_point", []), limit=2),
            "coils": coils_out,
            "flux_grid": {
                "nr": nr,
                "nz": nz,
                "psi_init": _clean_float_list(fg.get("psi_init", []), limit=nr * nz),
                "psi_opt": _clean_float_list(fg.get("psi_opt", []), limit=nr * nz),
                "psi_init_target": _json_scalar(fg.get("psi_init_target")),
                "psi_opt_target": _json_scalar(fg.get("psi_opt_target")),
            },
        }
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "field2d":
        panels_out = []
        for p in (raw.get("panels") or [])[:2]:
            if not isinstance(p, dict):
                continue
            nx = int(p.get("nx", 0))
            nz = int(p.get("nz", 0))
            vals = _clean_float_list(p.get("vals", []), limit=1024)
            if nx >= 2 and nz >= 2 and len(vals) == nx * nz:
                panels_out.append({
                    "label": str(p.get("label", "field"))[:40],
                    "nx": nx,
                    "nz": nz,
                    "vals": vals,
                    "vmin": _json_scalar(p.get("vmin")),
                    "vmax": _json_scalar(p.get("vmax")),
                })
        if not panels_out:
            return None
        out = {
            "kind": "field2d",
            "title": str(raw.get("title", "2D Field"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "panels": panels_out,
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "band1d":
        xg = _clean_float_list(raw.get("x_grid", []), limit=160)
        mu = _clean_float_list(raw.get("mu", []), limit=160)
        if not xg or len(xg) != len(mu):
            return None
        out = {
            "kind": "band1d",
            "title": str(raw.get("title", "Surrogate Fit & Uncertainty"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "x_grid": xg,
            "y_true": _clean_float_list(raw.get("y_true", []), limit=160),
            "mu": mu,
            "lo_2se": _clean_float_list(raw.get("lo_2se", []), limit=160),
            "hi_2se": _clean_float_list(raw.get("hi_2se", []), limit=160),
            "mw_lo": _clean_float_list(raw.get("mw_lo", []), limit=160),
            "mw_hi": _clean_float_list(raw.get("mw_hi", []), limit=160),
            "train_x": _clean_float_list(raw.get("train_x", []), limit=128),
            "train_y": _clean_float_list(raw.get("train_y", []), limit=128),
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    return None


def _summarize_state(ns):
    opt = None
    opt_name = None
    all_opts = []
    seen_ids = set()
    OptimizerState = getattr(baux, "OptimizerState", None)
    if OptimizerState is not None:
        for k, v in ns.items():
            if k.startswith("_") or not isinstance(v, OptimizerState):
                continue
            if int(getattr(v, "num_observations", 0)) <= 0 and k != "state":
                continue
            oid = id(v)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            all_opts.append((k, v))
        if isinstance(ns.get("state"), OptimizerState):
            opt = ns["state"]
            opt_name = "state"
        elif all_opts:
            opt_name, opt = all_opts[-1]
        else:
            for k in reversed(list(ns.keys())):
                if not k.startswith("_") and isinstance(ns[k], OptimizerState):
                    opt = ns[k]
                    opt_name = k
                    break

    fig_out = _extract_fig(ns.get("fig"))
    if opt is None and fig_out is None:
        return None

    summary = {"fig": fig_out}
    if opt is not None:
        y_raw = getattr(opt, "y_obs", None)
        x_raw = getattr(opt, "x_obs", None)
        y_list = []
        best_trace = []
        recent_rows = []
        if y_raw is not None:
            ya = np.asarray(y_raw, dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] > 0:
                prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
                running = -math.inf
                for v in prim.tolist():
                    fv = float(v)
                    if fv > running:
                        running = fv
                    y_list.append(fv if math.isfinite(fv) else None)
                    best_trace.append(running if math.isfinite(running) else None)
                if x_raw is not None:
                    xa = np.asarray(x_raw, dtype=float)
                    start = max(0, xa.shape[0] - 8)
                    for idx in range(start, xa.shape[0]):
                        row_x = [round(float(z), 4) for z in xa[idx, :6].tolist()]
                        recent_rows.append({
                            "obs": idx + 1,
                            "y": round(float(prim[idx]), 6),
                            "x": row_x,
                            "best_so_far": round(float(best_trace[idx]), 6) if best_trace[idx] is not None else None,
                        })

        best_x_val = getattr(opt, "best_x", None)
        best_x_list = None
        if best_x_val is not None:
            arr = np.asarray(best_x_val, dtype=float).ravel()
            best_x_list = [round(float(z), 5) for z in arr[:8].tolist()]

        best_params_repr = None
        bp = getattr(opt, "best_params", None)
        if bp is not None:
            try:
                best_params_repr = repr(bp)
                if len(best_params_repr) > 180:
                    best_params_repr = best_params_repr[:177] + "..."
            except Exception:
                best_params_repr = None

        summary.update({
            "var": opt_name,
            "dim": int(opt.dim),
            "epoch": int(opt.epoch),
            "num_observations": int(opt.num_observations),
            "num_regions": int(opt.num_regions),
            "tr_length": _json_scalar(opt.tr_length),
            "best_y": _json_scalar(opt.best_y),
            "best_x": best_x_list,
            "best_params": best_params_repr,
            "pending_count": 0 if opt.pending_ids is None else len(opt.pending_ids),
            "y_history": y_list[-128:],
            "best_history": best_trace[-128:],
            "recent": recent_rows,
        })

    if fig_out is None and len(all_opts) >= 2:
        auto_series = []
        for name, item in all_opts[:5]:
            ya = np.asarray(getattr(item, "y_obs", None), dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] <= 0:
                continue
            prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
            running = -math.inf
            bt = []
            for v in prim.tolist():
                fv = float(v)
                if fv > running:
                    running = fv
                if math.isfinite(running):
                    bt.append(round(running, 6))
            if bt:
                auto_series.append({
                    "label": str(name)[:40],
                    "x": list(range(1, len(bt) + 1))[-128:],
                    "y": bt[-128:],
                    "dashed": False,
                })
        if len(auto_series) >= 2:
            summary["fig"] = {
                "kind": "curves",
                "title": "Cumulative Best Objective Comparison",
                "subtitle": "Auto-detected OptimizerState trajectories in workspace",
                "x_label": "Evaluations",
                "y_label": "Best Objective",
                "log_y": False,
                "series": auto_series,
            }

    return summary


def _baux_exec_script(src, reset_ns=False):
    if reset_ns:
        _reset_namespace()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    try:
        tree = ast.parse(src, filename="<playground>", mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body.pop().value)
            if tree.body:
                exec(compile(tree, "<playground>", "exec"), _REPL_NS)
            val = eval(compile(last_expr, "<playground>", "eval"), _REPL_NS)
            if val is not None:
                print(repr(val))
        else:
            exec(compile(tree, "<playground>", "exec"), _REPL_NS)
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    summary = _summarize_state(_REPL_NS)
    return json.dumps({
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "incomplete": False,
        "elapsed_ms": round(dt_ms, 2),
        "state": summary,
    })


def _baux_repl_push(line):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    incomplete = False
    try:
        incomplete = bool(_CONSOLE.push(line))
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    err_txt = err_buf.getvalue()
    if err_txt and not incomplete:
        ok = False
    summary = _summarize_state(_REPL_NS)
    return json.dumps({
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_txt,
        "incomplete": incomplete,
        "elapsed_ms": round(dt_ms, 2),
        "state": summary,
    })


def _baux_complete(prefix):
    comp = rlcompleter.Completer(_REPL_NS)
    items = []
    idx = 0
    while len(items) < 24:
        cand = comp.complete(prefix, idx)
        if cand is None:
            break
        items.append(cand)
        idx += 1
    return json.dumps(items)
`;

async function ensureRuntime() {
  if (runtimePromise !== null) {
    return runtimePromise;
  }
  runtimePromise = (async () => {
    self.postMessage({ type: "status", stage: "downloading_manifest", detail: "Fetching Python runtime manifest..." });
    const manifest = await fetchJsonStrict("./repl_manifest.json");
    const zipSpec = manifest.baux_py_zip;

    self.postMessage({ type: "status", stage: "downloading_wasm", detail: "Loading CPython 3.12 WebAssembly kernel..." });
    const [pyodide, zipBuf] = await Promise.all([
      loadPyodide({ indexURL: "./pyodide/" }),
      fetchVerifiedBytes("./" + zipSpec.path, zipSpec.size, zipSpec.sha256),
    ]);

    self.postMessage({ type: "status", stage: "loading_numpy", detail: "Loading NumPy WebAssembly package..." });
    await pyodide.loadPackage(["numpy"]);

    self.postMessage({ type: "status", stage: "mounting_baux", detail: "Verifying & mounting baux Python package..." });
    pyodide.FS.mkdirTree("/home/pyodide/baux_pkg");
    pyodide.unpackArchive(zipBuf, "zip", { extractDir: "/home/pyodide/baux_pkg" });
    try {
      const envResp = await fetch("./baux_env.json", { credentials: "same-origin" });
      if (envResp.ok) {
        const envBuf = await envResp.arrayBuffer();
        pyodide.FS.writeFile("/home/pyodide/baux_env.json", new Uint8Array(envBuf));
      }
    } catch (_) {
      // Optional bundled physics metadata; offline or minimal bundles may omit it.
    }
    await pyodide.runPythonAsync(BOOTSTRAP_PY);

    self.postMessage({ type: "status", stage: "ready", detail: "Python 3.12 + NumPy + BAUX ready" });
    return pyodide;
  })().catch((err) => {
    runtimePromise = null;
    throw err;
  });
  return runtimePromise;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const id = msg.id || null;
  try {
    if (msg.type === "init") {
      await ensureRuntime();
      self.postMessage({ type: "result", id, ok: true, stdout: "", stderr: "", incomplete: false });
      return;
    }

    const pyodide = await ensureRuntime();

    if (msg.type === "reset_ns") {
      pyodide.globals.get("_reset_namespace")();
      self.postMessage({ type: "result", id, ok: true, stdout: "", stderr: "", incomplete: false, state: null });
      return;
    }

    if (msg.type === "exec") {
      const fn = pyodide.globals.get("_baux_exec_script");
      const raw = fn(String(msg.code || ""), Boolean(msg.resetNamespace));
      const payload = JSON.parse(raw);
      self.postMessage(Object.assign({ type: "result", id }, payload));
      return;
    }

    if (msg.type === "repl") {
      const fn = pyodide.globals.get("_baux_repl_push");
      const raw = fn(String(msg.line || ""));
      const payload = JSON.parse(raw);
      self.postMessage(Object.assign({ type: "result", id }, payload));
      return;
    }

    if (msg.type === "complete") {
      const fn = pyodide.globals.get("_baux_complete");
      const raw = fn(String(msg.prefix || ""));
      self.postMessage({ type: "complete", id, items: JSON.parse(raw) });
      return;
    }

    self.postMessage({ type: "result", id, ok: false, stdout: "", stderr: "Unknown worker command", incomplete: false });
  } catch (err) {
    self.postMessage({
      type: "result",
      id,
      ok: false,
      stdout: "",
      stderr: err && err.stack ? String(err.stack) : String(err),
      incomplete: false,
    });
  }
};
