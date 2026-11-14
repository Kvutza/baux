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


def _summarize_state(ns):
    opt = None
    opt_name = None
    OptimizerState = getattr(baux, "OptimizerState", None)
    if OptimizerState is not None:
        if isinstance(ns.get("state"), OptimizerState):
            opt = ns["state"]
            opt_name = "state"
        else:
            for k in reversed(list(ns.keys())):
                if k.startswith("_"):
                    continue
                if isinstance(ns[k], OptimizerState):
                    opt = ns[k]
                    opt_name = k
                    break
    if opt is None:
        return None

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
            for i, v in enumerate(prim.tolist()):
                fv = float(v)
                if fv > running:
                    running = fv
                y_list.append(fv if math.isfinite(fv) else None)
                best_trace.append(running if math.isfinite(running) else None)
            if x_raw is not None:
                xa = np.asarray(x_raw, dtype=float)
                start = max(0, xa.shape[0] - 10)
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

    return {
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
    }


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
